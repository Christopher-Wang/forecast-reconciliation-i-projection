from enum import auto
from typing import ClassVar

import numpy as np
import pandas as pd
import torch

from src.datasets_manager import PerLevelTestValPreds
from src.reconciliation.common import Reconciler, ReconcilerName, ReconciledRes
from src.utils.general_utils import ListableEnum


class IProjRes(ReconciledRes):
    rec_probs: torch.Tensor
    rec_means: torch.Tensor
    lambdas: torch.Tensor
    residual: torch.Tensor
    iters: torch.Tensor
    converged: torch.Tensor

    model_config = {"arbitrary_types_allowed": True}

    def get_reconciled_means(self) -> np.ndarray:
        return self.rec_means.swapaxes(0, 1).flatten().detach().cpu().numpy()


class IProjWeightStrategy(ListableEnum):
    UNIFORM = auto()
    SQRT_SIG_TO_NOISE = auto()
    STRUCT = auto()
    TARGET_MATCHING = auto()


class IProjReconciler(Reconciler):
    proj_strat: IProjWeightStrategy
    name: ClassVar[ReconcilerName] = ReconcilerName.I_PROJ
    max_iter: int = 100
    tol: float = 1e-10
    ridge: float = 2e-8
    max_halving: int = 20

    def get_fullname(self) -> str:
        match self.proj_strat:
            case IProjWeightStrategy.UNIFORM:
                return self.name.value
            case IProjWeightStrategy.SQRT_SIG_TO_NOISE:
                return self.name.value + "-W"
            case IProjWeightStrategy.STRUCT:
                return self.name.value + "-STRUCT"
            case IProjWeightStrategy.TARGET_MATCHING:
                return self.name.value + "-TM"
            case _:
                raise ValueError

    def get_color(self, ) -> str:
        match self.proj_strat:
            case IProjWeightStrategy.UNIFORM:
                return "#1f77b4"
            case IProjWeightStrategy.SQRT_SIG_TO_NOISE:
                return "#bcbd22"
            case IProjWeightStrategy.STRUCT:
                return "#17becf"
            case _:
                raise ValueError

    def perform_projection(
            self, per_level_test_val_preds: PerLevelTestValPreds, aggregation_matrix: pd.DataFrame,
            tags: dict[str, pd.DataFrame], target_means: torch.Tensor | None = None,
    ) -> IProjRes:
        per_level_test_logits, per_level_val_test_logits = [], []
        per_level_test_bucket_means = []
        for level in aggregation_matrix.index.unique():
            test_pred = per_level_test_val_preds.test_preds[level]
            val_test_pred = per_level_test_val_preds.val_test_preds[level]

            per_level_test_logits.append(test_pred.pred_distribution.logits)
            per_level_val_test_logits.append(val_test_pred.pred_distribution.logits)
            per_level_test_bucket_means.append(test_pred.pred_distribution.bucket_means)

        n_total_ts, n_base_ts = aggregation_matrix.shape
        h_constr = np.hstack([np.eye(n_total_ts - n_base_ts), -aggregation_matrix.values[: n_total_ts - n_base_ts, :]])

        per_level_test_logits = torch.stack(per_level_test_logits, dim=0).swapaxes(0, 1)
        per_level_test_logits = torch.where(per_level_test_logits == -np.inf, -30, per_level_test_logits)
        h_constr = torch.Tensor(h_constr).to(per_level_test_logits.device)

        per_level_test_bucket_means = torch.stack(per_level_test_bucket_means, dim=0)[None, :, :]

        match self.proj_strat:
            case IProjWeightStrategy.UNIFORM:
                w = None
                reconciled = self.kl_expectation_projection_dual_newton_batched(
                    base_logits=per_level_test_logits, v=per_level_test_bucket_means, h_constr=h_constr,
                    series_weights=w
                )

            case IProjWeightStrategy.SQRT_SIG_TO_NOISE:
                val_test_df_copy = per_level_test_val_preds.val_test_df.copy()
                val_test_df_copy['errors'] = (
                        val_test_df_copy['y'] - per_level_test_val_preds.flatten_val_test_pred_means)
                noise = val_test_df_copy.groupby('unique_id')['errors'].var().loc[aggregation_matrix.index].values
                signal = val_test_df_copy.groupby('unique_id')['y'].var().loc[aggregation_matrix.index].values
                w = (signal / noise)
                w = w.clip(min=w[w > 0].min())
                # m = 1  # seasonality lag for naive forecast

                # df = per_level_test_val_preds.val_test_df.copy()
                # df["abs_err"] = (df["y"] - per_level_test_val_preds.flatten_val_test_pred_means).abs()

                # mae   = df.groupby("unique_id")["abs_err"].mean()
                # denom = (df.sort_values(["unique_id", "ds"])
                #         .groupby("unique_id")["y"]
                #         .apply(lambda s: (s - s.shift(m)).abs().dropna().mean()))

                # mase = (mae / denom).reindex(aggregation_matrix.index)
                # eps  = mase[mase > 0].min()
                # w    = mase.fillna(eps).clip(lower=eps).values +100
                match self.proj_strat:
                    case IProjWeightStrategy.SQRT_SIG_TO_NOISE:
                        beta = 0.5
                    case _:
                        raise ValueError
                w = torch.Tensor(w).to(per_level_test_logits.device) ** beta
                reconciled = self.kl_expectation_projection_dual_newton_batched(
                    base_logits=per_level_test_logits, v=per_level_test_bucket_means, h_constr=h_constr,
                    series_weights=w
                )

            case IProjWeightStrategy.STRUCT:
                l = aggregation_matrix.values @ aggregation_matrix.values.T
                w = torch.Tensor(np.diagonal(l)).to(per_level_test_logits.device)
                reconciled = self.kl_expectation_projection_dual_newton_batched(
                    base_logits=per_level_test_logits, v=per_level_test_bucket_means, h_constr=h_constr, 
                    series_weights=w
                )

            case _:
                raise ValueError
        return reconciled

    @torch.no_grad()
    def kl_expectation_projection_dual_newton_batched(
            self,
            base_logits: torch.Tensor,
            v: torch.Tensor,
            h_constr: torch.Tensor,  # H
            b: torch.Tensor | None = None,
            series_weights: torch.Tensor | None = None,
            dtype: torch.dtype = torch.float64,
            device: torch.device | None = None,
    ) -> IProjRes:
        """
        Batched damped Dual Newton for:
            minimize  Σ_i w_i * KL(p_i || q_i)   (per batch)
            subject to H μ(p) = b,
        with μ_i = ⟨v_i, p_i⟩ and
            p_i ∝ q_i * exp( - ((H^T λ)_i / w_i) * v_i ).

        Notes:
        - w_i > 0 are per-series KL weights: larger w_i ⇒ smaller tilt ⇒ that series moves less.
        - Dual residual g(λ) = H μ(λ) - b.
        - Dual “Hessian”:  -∇g(λ) = H diag(Var_{p_i}(v_i)/w_i) H^T  (SPD).

        Args:
            base_logits: batch of base logits
            v:
            h_constr:
            b:
            series_weights:
            dtype:
            device:

        Returns:
            Result of the I-projection
        """
        x = base_logits.to(device=device, dtype=dtype).contiguous()
        batch_size, n, k = x.shape

        v = v.to(device=device, dtype=dtype)
        if v.dim() == 1:
            assert v.shape[0] == k
            v_b = v.view(1, 1, k).expand(batch_size, n, k)
        elif v.dim() == 3:
            assert v.shape[1] == n and v.shape[2] == k
            v_b = v.expand(batch_size, n, k) if v.shape[0] == 1 else v.contiguous()
        else:
            raise ValueError("v must be (k,) or (1,n,k) or (B,n,k)")

        h_constr = h_constr.to(device=device, dtype=dtype)
        if h_constr.dim() == 2:
            r, _ = h_constr.shape
            assert h_constr.shape[1] == n, (h_constr.shape, n)
            h_constr_batch = h_constr.view(1, r, n).expand(batch_size, r, n).contiguous()
        elif h_constr.dim() == 3:
            assert h_constr.shape[2] == n and h_constr.shape[0] in (1, batch_size)
            r = h_constr.shape[1]
            h_constr_batch = h_constr.expand(batch_size, -1, -1) if h_constr.shape[0] == 1 else h_constr.contiguous()
        else:
            raise ValueError("H must be (r, n) or (batch_size, r, n)")

        if b is None:
            b_b = torch.zeros(batch_size, r, dtype=dtype, device=x.device)
        else:
            b = b.to(device=device, dtype=dtype)
            if b.dim() == 1:
                assert b.shape[0] == r
                b_b = b.view(1, r).expand(batch_size, r).contiguous()
            elif b.dim() == 2:
                assert b.shape[1] == r and b.shape[0] in (1, batch_size)
                b_b = b.expand(batch_size, r) if b.shape[0] == 1 else b.contiguous()
            else:
                raise ValueError("b must be (r,) or (B,r)")

        if series_weights is None:
            w = torch.ones(batch_size, n, dtype=dtype, device=x.device)
        else:
            sw = series_weights.to(device=device, dtype=dtype)
            if sw.dim() == 1:
                assert sw.shape[0] == n
                w = sw.view(1, n).expand(batch_size, n).contiguous()
            elif sw.dim() == 2:
                assert sw.shape[1] == n and sw.shape[0] in (1, batch_size)
                w = sw.expand(batch_size, n) if sw.shape[0] == 1 else sw.contiguous()
            else:
                raise ValueError("series_weights must be (n,) or (B,n)")
            if torch.any(w <= 0):
                raise ValueError("series_weights must be positive.")

        identity_r = torch.eye(r, dtype=dtype, device=x.device).expand(batch_size, r, r)

        def _p_mu_var(lam_: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            theta = torch.einsum('brn,br->bn', h_constr_batch, lam_)
            log_p = x - (theta / w).unsqueeze(-1) * v_b
            p_ = torch.softmax(log_p, dim=-1)
            mu_ = (p_ * v_b).sum(dim=-1)
            ev2 = (p_ * (v_b * v_b)).sum(dim=-1)
            var_ = torch.clamp(ev2 - mu_ * mu_, min=1e-18)
            return p_, mu_, var_

        lam = torch.zeros(batch_size, r, dtype=dtype, device=x.device)
        p, mu, var = _p_mu_var(lam)
        resid = torch.einsum('brn,bn->br', h_constr_batch, mu) - b_b
        best_norm = resid.abs().amax(dim=1)

        it = 0
        for it in range(self.max_iter):
            converged_mask = best_norm <= self.tol
            if torch.all(converged_mask):
                break

            hessian_cols = h_constr_batch * (var / w).unsqueeze(1)
            a_mat = torch.matmul(hessian_cols, h_constr_batch.transpose(-1, -2)) + self.ridge * identity_r
            try:
                chol_a_mat = torch.linalg.cholesky(a_mat)
                delta = torch.cholesky_solve(resid.unsqueeze(-1), chol_a_mat).squeeze(-1)
            except RuntimeError:
                delta = torch.linalg.solve(a_mat, resid)

            step = torch.ones(batch_size, dtype=dtype, device=x.device)
            active_mask = ~converged_mask

            for _ in range(self.max_halving):
                if not active_mask.any():  # Stop if all active batches have improved
                    break

                # Try the step for active batches
                lam_try = lam.clone()
                lam_try[active_mask] = lam[active_mask] + step[active_mask].unsqueeze(-1) * delta[active_mask]

                _, mu_try, _ = _p_mu_var(lam_try)
                resid_try = torch.einsum('brn,bn->br', h_constr_batch, mu_try) - b_b
                new_norm = resid_try.abs().amax(dim=1)

                # Check for improvement only on active batches
                improved_mask = new_norm < best_norm

                # Batches that just improved are no longer active in the line search
                newly_improved_mask = improved_mask & active_mask

                # Update state for batches that just improved
                lam[newly_improved_mask] = lam_try[newly_improved_mask]
                resid[newly_improved_mask] = resid_try[newly_improved_mask]
                best_norm[newly_improved_mask] = new_norm[newly_improved_mask]

                # Update the active mask for the next halving step
                active_mask = active_mask & ~improved_mask

                # Halve the step size for batches that are still active (i.e., failed to improve)
                step[active_mask] *= 0.5

            p, mu, var = _p_mu_var(lam)

        it_used = it + 1 if it < self.max_iter - 1 else self.max_iter
        converged = best_norm <= self.tol
        return IProjRes(
            rec_probs=p, rec_means=mu, lambdas=lam, residual=resid,
            iters=torch.full((batch_size,), it_used, dtype=torch.int64, device="cpu"),
            converged=converged
        )
