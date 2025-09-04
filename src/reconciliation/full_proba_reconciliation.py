from __future__ import annotations

import enum

import numpy as np
import torch
from pydantic import BaseModel, field_validator, model_validator, PrivateAttr
from tqdm import tqdm
from typing_extensions import Self

from src.utils.predictions_utils import RiemannDistrib, PFNPreds, DeviceLikeType


def generate_sum_bucket_bounds(
        mean_1: float, mean_2: float, std_1: float, std_2: float, standard_boundaries: torch.Tensor
) -> torch.Tensor:
    """

    Args:
        mean_1:
        mean_2:
        std_1:
        std_2:
        standard_boundaries:

    Returns:
        new bucket bounds
    """
    # Linearity of Variance for independence
    mean_1_2 = mean_1 + mean_2
    std_1_2 = (std_1 ** 2 + std_2 ** 2) ** 0.5
    bucket_bounds_1_2 = standard_boundaries * std_1_2 + mean_1_2
    return bucket_bounds_1_2


def get_sum_of_two_dists(
        prob_1, prob_2, bucket_means_1, bucket_means_2, target_bucket_bounds: torch.Tensor
) -> torch.Tensor:
    """
    Compute the distribution of X_1 + X_2 assuming they are independent and characterized by Riemann distributions.

    Args:
        prob_1: proba on each bucket of X1 (batch, n_bins)
        prob_2: proba on each bucket of X2 (batch, n_bins)
        bucket_means_1: buckets mean values of X1 distribution (n_bins,)
        bucket_means_2: buckets mean values of X2 distribution (n_bins,)
        target_bucket_bounds: target bucket boundaries of X1 + X2 distribution (n_bins + 1,)

    Returns:
        probs of distrib of X1 + X2: (batch, n_bins)
    """
    assert prob_1.shape == prob_2.shape, (prob_1.shape, prob_2.shape)
    prob_1_2 = (prob_1[:, :, None] * prob_2[:, None, :]).flatten(start_dim=1)
    bucket_bounds_1_2 = (bucket_means_1[:, None] + bucket_means_2[None, :]).flatten(start_dim=0)
    index_1_2 = (torch.bucketize(bucket_bounds_1_2, target_bucket_bounds) - 1).clip(min=0, max=prob_1.shape[-1] - 1)
    output = torch.zeros(prob_1.shape).to(prob_1.device)
    output = output.index_add(1, index_1_2, prob_1_2)
    return output


def sum_two_preds(
        preds: list[PFNPreds], target_bucket_bounds: torch.Tensor | None,
        standard_boundaries: torch.Tensor | None, device: DeviceLikeType
) -> PFNPreds:
    """

    Args:
        preds:
        device:
        target_bucket_bounds:
        standard_boundaries: if target_bucket_bounds is not specified, this should be

    Returns:

    """
    assert target_bucket_bounds is not None or standard_boundaries is not None
    assert len(preds) == 2, len(preds)

    compute_preds: list[PFNPreds] = []
    for i in range(2):
        compute_preds.append(preds[i].to(device=device))

    train_sum_mean = compute_preds[0].train_mean + compute_preds[1].train_mean
    train_sum_std = (compute_preds[0].train_std ** 2 + compute_preds[1].train_std ** 2) ** 0.5

    if target_bucket_bounds is None:
        target_bucket_bounds = generate_sum_bucket_bounds(
            mean_1=compute_preds[0].train_mean, mean_2=compute_preds[1].train_mean,
            std_1=compute_preds[0].train_std, std_2=compute_preds[1].train_std,
            standard_boundaries=standard_boundaries
        )
    target_bucket_bounds = target_bucket_bounds.to(device=device)

    probs_1_2 = get_sum_of_two_dists(
        prob_1=compute_preds[0].pred_distribution.logits.softmax(-1),
        prob_2=compute_preds[1].pred_distribution.logits.softmax(-1),
        bucket_means_1=compute_preds[0].pred_distribution.bucket_means,
        bucket_means_2=compute_preds[1].pred_distribution.bucket_means,
        target_bucket_bounds=target_bucket_bounds
    )

    return PFNPreds(
        name=f"{preds[0].name} + {preds[1].name}",
        pred_distribution=RiemannDistrib(bucket_borders=target_bucket_bounds, logits=(probs_1_2 + 1e-13).log()),
        train_mean=train_sum_mean, train_std=train_sum_std
    )


def sum_n_preds(
        preds: list[PFNPreds], target_bucket_bounds: torch.Tensor | None,
        standard_boundaries: torch.Tensor | None, device: DeviceLikeType
) -> PFNPreds:
    # TODO: improve sorting
    current_base_preds = sorted(preds, key=lambda pfn_pred: pfn_pred.train_mean)
    while len(current_base_preds) > 1:
        for i in range(len(current_base_preds) // 2):
            current_base_preds[i] = sum_two_preds(
                preds=[current_base_preds[2 * i], current_base_preds[2 * i + 1]],
                target_bucket_bounds=target_bucket_bounds, standard_boundaries=standard_boundaries, device=device
            )
            # gpu_usage = get_gpu_usage(device=device)
            # print("GPU usage:", gpu_usage)
            # if gpu_usage > .5:
            #     for j in range(i + 1):
            #         current_base_preds[j] = current_base_preds[j].to(device="cpu")
            #     torch.cuda.empty_cache()
            #     gpu_usage = get_gpu_usage(device=device)
            #     print("GPU usage after empty cache:", gpu_usage)

        current_base_preds = current_base_preds[:(len(current_base_preds) // 2 + len(current_base_preds) % 2)]
        current_base_preds = sorted(current_base_preds, key=lambda pfn_pred: pfn_pred.train_mean)
    return current_base_preds[0]


def emd_1d_discrete(probs_1: torch.Tensor, probs_2: torch.Tensor, bin_means: torch.Tensor) -> torch.Tensor:
    """
    Compute 1D Earth Mover's Distance for discrete distributions.

    Args:
        probs_1: Tensor of shape (batch, n_bins) — probabilities (non-negative, sum to 1).
        probs_2: Tensor of shape (batch, n_bins) — probabilities (non-negative, sum to 1).
        bin_means: Tensor of shape (n_bins,) — sorted support points.

    Returns:
        Tensor of shape (n_bins,) of EMD values
    """

    # Cumulative difference
    cdf_diff = torch.cumsum(probs_1 - probs_2, dim=-1)

    # Distances between successive x
    dx = bin_means[1:] - bin_means[:-1]

    # Contribution is |cdf_diff| * dx for each interval
    emd = torch.sum(torch.abs(cdf_diff[:, :-1]) * dx[None, :], dim=-1)
    return emd


class LossType(enum.Enum):
    KL = "KL"
    EMD = "EMD"


class FullProbaPairwiseReconciler(BaseModel):
    all_preds: dict[str, PFNPreds]
    top_level_name: str

    _saved_preds: dict[str, PFNPreds] = PrivateAttr()

    model_config = {"arbitrary_types_allowed": True}

    def model_post_init(self, __context) -> None:
        self._saved_preds = {name: pred.to(device=torch.device("cpu")) for name, pred in self.all_preds.items()}

    @property
    def saved_preds(self) -> dict[str, PFNPreds]:
        """Expose read-only access to the saved initial predictions."""
        return self._saved_preds

    def get_target_pred(self) -> PFNPreds:
        return self.all_preds[self.top_level_name]

    @field_validator("all_preds", mode="after")
    @classmethod
    def consistent_name(cls, value: dict[str, PFNPreds]) -> dict[str, PFNPreds]:
        for name, pred in value.items():
            if pred.name != name:
                raise ValueError(f"Expected {pred.name} but got it under name {name}")
        return value

    @model_validator(mode='after')
    def check_top_level_name(self) -> Self:
        if self.top_level_name not in self.all_preds:
            raise ValueError(f"{self.top_level_name} not in {self.all_preds.keys()}")
        return self

    @field_validator("all_preds", mode="before")
    def skip_validation_if_instance(cls, v: dict[str, PFNPreds]) -> dict[str, PFNPreds]:
        if isinstance(v, dict) and all(isinstance(p, PFNPreds) for p in v.values()):
            return v
        for name, pred in v.items():
            print(name, type(pred))
        raise ValueError(f"{cls.__name__} must return dict with a PFNPreds instance")

    def reconcile_sgd(
            self, lr: float, num_steps: int, device: DeviceLikeType, ground_truth: dict[str, np.ndarray],
            loss_type: LossType
    ) -> None:

        opt = torch.optim.SGD(
            (pred.pred_distribution.logits.requires_grad_(True) for name, pred in self.all_preds.items() if
             name != self.top_level_name),
            lr=lr, momentum=0.9, nesterov=True
        )
        eps = 1e-13

        original_log_prob_preds = {}
        orginal_prob_preds = {}
        for name, pred in self._saved_preds.items():
            original_prob_pred = pred.pred_distribution.logits.detach().to(device=device).softmax(-1)
            original_log_prob_preds[name] = (
                ((original_prob_pred + eps) / (1 + eps * original_prob_pred.shape[-1])).log()).to("cpu")
            orginal_prob_preds[name] = original_prob_pred.to("cpu")

        sum_mae = 0
        n_cats = 0
        for name in ground_truth:
            if name == self.top_level_name:
                continue
            sum_mae += np.abs(
                self.all_preds[name].mean_pred.detach().cpu().numpy().flatten() - ground_truth[name]
            ).mean()
            n_cats += 1
        print(f">>> Original MAE: {sum_mae / n_cats:g}")

        losses = []
        for _ in tqdm(range(num_steps)):
            # zero the gradients
            opt.zero_grad()

            base_preds = [pred for name, pred in self.all_preds.items() if name != self.top_level_name]
            reconciled_total = sum_n_preds(
                preds=base_preds,
                target_bucket_bounds=self.get_target_pred().pred_distribution.bucket_borders,
                standard_boundaries=None,
                device=device
            )
            base_losses = 0
            for name, pred in self.all_preds.items():
                if name == self.top_level_name:
                    continue
                logits = pred.pred_distribution.logits
                pred_probs = logits.softmax(-1).to(device=device)
                if loss_type == LossType.KL:
                    base_losses += torch.sum(
                        pred_probs * (
                                ((pred_probs + eps) / (1 + eps * pred_probs.shape[-1])).log() - original_log_prob_preds[
                            name].to(pred_probs))
                    )
                elif loss_type == LossType.EMD:
                    base_losses += torch.sum(
                        emd_1d_discrete(
                            probs_1=pred_probs, probs_2=orginal_prob_preds[name].to(pred_probs),
                            bin_means=pred.pred_distribution.bucket_means.to(pred_probs)
                        )
                    )

            pred_probs = reconciled_total.pred_distribution.logits.softmax(-1)
            if loss_type == LossType.KL:
                top_loss = torch.sum(
                    pred_probs * (
                            ((pred_probs + eps) / (1 + eps * pred_probs.shape[-1])).log() - original_log_prob_preds[
                        self.top_level_name].to(pred_probs))
                )
            elif loss_type == LossType.EMD:
                top_loss = torch.sum(
                    emd_1d_discrete(
                        probs_1=pred_probs, probs_2=orginal_prob_preds[self.top_level_name].to(pred_probs),
                        bin_means=pred.pred_distribution.bucket_means.to(pred_probs)
                    )
                )

            losses.append(top_loss.item())
            print(
                f">>> Base loss {base_losses.item():g} | Top loss {top_loss.item():g} | Sum loss {base_losses.item() + top_loss.item():g}"
                )
            sum_mae = 0
            n_cats = 0
            for name in ground_truth:
                if name == self.top_level_name:
                    continue
                sum_mae += np.abs(
                    self.all_preds[name].mean_pred.detach().cpu().numpy().flatten() - ground_truth[name]
                ).mean()
                n_cats += 1
            print(f">>> MAE: {sum_mae / n_cats:g}")

            (top_loss + base_losses).backward()
            # top_loss.backward()
            opt.step()
