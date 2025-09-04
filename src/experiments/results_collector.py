from __future__ import annotations

from enum import auto
from typing import Callable
import numpy as np
import pandas as pd
import torch
from hierarchicalforecast.evaluation import evaluate
from pydantic import BaseModel
from tqdm import tqdm
from utilsforecast.losses import rmse, mae

from src.datasets_manager import HierarchicalDataset, HierarchicalDatasetNames
from src.reconciliation.common import NoReconciler, CLASSIC_RECONCILERS, MinTReconciler, MinTMethodName
from src.reconciliation.i_projection import IProjReconciler, IProjWeightStrategy
from src.utils.general_utils import ListableEnum


class MetricName(ListableEnum):
    RMSE = "rmse"
    MAE = "mae"

    @staticmethod
    def get_forecast_loss(metric_name: MetricName) -> Callable:
        match metric_name:
            case MetricName.RMSE:
                return rmse
            case MetricName.MAE:
                return mae
            case _:
                raise ValueError


class ResultKey(ListableEnum):
    def _generate_next_value_(name, start: ..., count: int, last_values: ...) -> str:
        """ Generate the next value when not given. """
        return name.lower()

    DATASET = auto()
    HORIZON = auto()
    FOLD = auto()
    LEVEL = auto()
    METHOD = auto()
    METRIC = auto()
    VALUE = auto()
    METHOD_COLOR = auto()

import torch

def quantile_calibration_error(
    y: torch.Tensor,        # (B,T)
    edges: torch.Tensor,    # (B,K+1)  per-series edges (monotone)
    probs: torch.Tensor,    # (B,T,K) or (B,K) -> broadcast over T
    eps: float = 1e-12
) -> torch.Tensor:
    """
    Quantile Calibration Error (QCE) per series.
    Returns: (B,), pooling across time T and averaging over a small tau grid.

    QCE_b = mean_tau | (1/T) * sum_t 1{ y_{b,t} <= q_{b,t}(tau) } - tau |
    """
    # modest tau grid for stability with small T
    taus = torch.tensor([0.10, 0.25, 0.50, 0.75, 0.90], dtype=y.dtype, device=y.device)  # (J,)
    J = taus.numel()

    if probs.dim() == 2:
        B, T = y.shape
        probs = probs[:, None, :].expand(B, T, probs.shape[-1])
    B, T, K = probs.shape
    if edges.shape != (B, K + 1):
        raise ValueError("`edges` must be (B, K+1) matching probs[..., K].")

    L = edges[:, :-1].to(y.dtype).to(y.device)          
    U = edges[:,  1:].to(y.dtype).to(y.device)          
    w = (U - L).clamp_min(eps)                            

    cum = probs.cumsum(dim=-1)                           

    taus_btj = taus.view(1, 1, J).expand(B, T, J)      
    m = torch.searchsorted(cum, taus_btj, right=False) 
    m = torch.clamp(m, max=K-1)

    mb1 = torch.clamp(m - 1, min=0)
    cum_before = torch.gather(cum, dim=-1, index=mb1)   
    cum_before = torch.where(m.eq(0), torch.zeros_like(cum_before), cum_before)

    p_m = torch.gather(probs, dim=-1, index=m).clamp_min(eps)
    LBTK = L.unsqueeze(1).expand(B, T, K)
    wBTK = w.unsqueeze(1).expand(B, T, K)
    L_m  = torch.gather(LBTK, dim=-1, index=m) 
    w_m  = torch.gather(wBTK, dim=-1, index=m) 

    frac = ((taus_btj - cum_before) / p_m).clamp(0.0, 1.0)
    q = L_m + w_m * frac   

    cov_bj = (y.unsqueeze(-1) <= q).to(y.dtype).mean(dim=1)

    qce_b = (cov_bj - taus.view(1, J)).abs().mean(dim=-1) 
    return qce_b


def crps_piecewise_uniform_probs_btk(
    y: torch.Tensor,        # (B,T)
    edges: torch.Tensor,    # (B,K+1)  per-series edges (monotone)
    probs: torch.Tensor,    # (B,T,K) or (B,K) -> broadcast over T
    eps: float = 1e-12
) -> torch.Tensor:
    """
    CRPS for regression-via-classification with per-series edges, given probabilities.
    Returns (B,T).
    """
    if probs.dim() == 2:
        probs = probs[:, None, :].expand(y.shape[0], y.shape[1], -1)
    B, T, K = probs.shape
    if edges.shape != (B, K+1):
        raise ValueError("edges must be (B, K+1) matching probs[..., K].")

    # Align dtype/device and (safe) renormalize
    y = y.to(probs); e = edges.to(probs)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(eps)

    L, U = e[:, :-1], e[:, 1:]                 # (B,K)
    w = (U - L).clamp_min(eps)                 # (B,K)

    a = probs.cumsum(dim=-1) - probs           # (B,T,K) CDF at left edge
    b = probs / w.unsqueeze(1)                 # (B,T,K) slope within bin

    yBT = y.unsqueeze(-1)                      # (B,T,1)
    Wl  = (yBT - L.unsqueeze(1)).clamp_min(0.0)
    Wl  = torch.minimum(Wl, w.unsqueeze(1))    # clamp to bin width

    # Left: ∫_0^{Wl} (a + b t)^2 dt
    I_left  = (a*a)*Wl + (a*b)*(Wl**2) + (b*b)*(Wl**3)/3.0

    # Right: ∫_{Wl}^{w} (a + b t - 1)^2 dt  (definite integral)
    A1 = a - 1.0
    wB1 = w.unsqueeze(1)
    I_right = ((A1*A1)*(wB1 - Wl)
               + (A1*b)*(wB1**2 - Wl**2)
               + (b*b)*(wB1**3 - Wl**3)/3.0)

    tails = (e[:, :1] - y).clamp_min(0.0) + (y - e[:, -1:]).clamp_min(0.0)  # (B,T)
    return (I_left + I_right).sum(dim=-1) + tails

def get_borders(per_level_test_val_preds: PerLevelTestValPreds, aggregation_matrix: pd.DataFrame) -> torch:
        borders = []
        for level in aggregation_matrix.index.unique():
            border = per_level_test_val_preds.test_preds[level].pred_distribution.full_support_bar.borders
            borders.append(border)
        borders = torch.stack(borders)
        return borders

class ResultsCollector(BaseModel):

    @staticmethod
    def collect_probablistic_results(
        dataset_name: HierarchicalDatasetNames, horizon: int, n_cross_val: int,
    ):
        """

        Args:
            dataset:
            horizon:
            n_cross_val:

        Returns:
            A dictionary mapping fold to table for probablistic metrics
        """
        hierarchical_dataset = HierarchicalDataset.get_hierarchical_dataset(dataset_name=dataset_name)
        cross_val_preds, aggregation_matrix, tags = hierarchical_dataset.get_cross_val_data(
            n_cross_val=n_cross_val, horizon=horizon, device="cpu"
        )
        reverse_tags = {i: k for k, v in tags.items() for i in v  }
        result_tables = []
        for cross_val_ind in range(n_cross_val):
            y_rec = cross_val_preds[cross_val_ind].test_df.copy()
            y_rec['level'] = y_rec['unique_id'].apply(lambda x: reverse_tags[x])
            per_level_test_val_preds = cross_val_preds[cross_val_ind]
            borders = get_borders(per_level_test_val_preds, aggregation_matrix)

            reconcilers = [NoReconciler(custom_name="Tab-PFN")]
            i_proj_strats = [
                IProjWeightStrategy.UNIFORM, IProjWeightStrategy.SQRT_SIG_TO_NOISE, IProjWeightStrategy.STRUCT
            ]
            reconcilers += [IProjReconciler(proj_strat=proj_strat) for proj_strat in i_proj_strats]

            melted_table = []
            for reconciler in tqdm(reconcilers, desc="Go through reconcile methods"):
                reconcile_res = reconciler.perform_projection(
                    per_level_test_val_preds=per_level_test_val_preds,
                    aggregation_matrix=aggregation_matrix,
                    tags=tags,
                )
                probs = reconcile_res.rec_probs.swapaxes(0, 1)
                y = y_rec['y'].values.reshape(probs.shape[:-1])
                y = torch.Tensor(y).to(probs.device)
                crps = crps_piecewise_uniform_probs_btk(y, borders, probs).detach().cpu().numpy().flatten()
                y_rec['crps'] = crps
                df = y_rec.groupby('level')['crps'].mean().reset_index().to_dict(orient='records')
                df.append ({'level': 'Overall', 'crps': crps.mean()})
                df = pd.DataFrame(df)
                df['value'] = df['crps']
                df = df.drop('crps', axis=1)
                df['method'] = reconciler.get_fullname()
                df['metric'] = 'crps'
                melted_table.append(df)

                qce = quantile_calibration_error(y, borders, probs).detach().cpu().numpy().flatten()
                df = pd.DataFrame({
                    'level': [reverse_tags[_id] for _id in aggregation_matrix.index.unique()],
                    'qce': qce
                })
                df = df.groupby('level')['qce'].mean().reset_index().to_dict(orient='records')
                df.append ({'level': 'Overall', 'qce': qce.mean()})
                df = pd.DataFrame(df)
                df['value']=df['qce']
                df = df.drop('qce', axis=1)
                df['method'] = reconciler.get_fullname()
                df['metric'] = 'qce'
                melted_table.append(df)

            melted_table = pd.concat(melted_table)            
            melted_table[ResultKey.DATASET.value] = hierarchical_dataset.name.value
            melted_table[ResultKey.HORIZON.value] = horizon
            melted_table[ResultKey.FOLD.value] = cross_val_ind
            result_tables.append(melted_table)
        result_tables = pd.concat(result_tables, ignore_index=True)
        return result_tables

    @staticmethod
    def collect_dataset_results(
            dataset_name: HierarchicalDatasetNames, horizon: int, n_cross_val: int, metric_names: list[MetricName]
    ) -> pd.DataFrame:
        """

        Args:
            dataset:
            horizon:
            n_cross_val:
            metric_names:

        Returns:
            A dictionary mapping fold to table
        """
        hierarchical_dataset = HierarchicalDataset.get_hierarchical_dataset(dataset_name=dataset_name)
        cross_val_preds, aggregation_matrix, tags = hierarchical_dataset.get_cross_val_data(
            n_cross_val=n_cross_val, horizon=horizon, device="cpu"
        )
        metrics = [MetricName.get_forecast_loss(metric_name=metric_name) for metric_name in metric_names]
        result_tables = []
        for cross_val_ind in range(n_cross_val):
            y_rec = cross_val_preds[cross_val_ind].test_df.copy()
            per_level_test_val_preds = cross_val_preds[cross_val_ind]

            reconcilers = [NoReconciler(custom_name="Tab-PFN")]
            reconcilers += [c for c in CLASSIC_RECONCILERS]
            i_proj_strats = [
                IProjWeightStrategy.UNIFORM, IProjWeightStrategy.SQRT_SIG_TO_NOISE, IProjWeightStrategy.STRUCT
            ]
            reconcilers += [IProjReconciler(proj_strat=proj_strat) for proj_strat in i_proj_strats]
            rec_name_to_color = {}
            for reconciler in tqdm(reconcilers, desc="Go through reconcile methods"):
                if isinstance(
                        reconciler, IProjReconciler
                ) and reconciler.proj_strat == IProjWeightStrategy.TARGET_MATCHING:
                    target_reconciler = MinTReconciler(method=MinTMethodName.MINT_SHRINK, nonnegative=True)
                    c = y_rec[target_reconciler.get_fullname()].values.astype(np.float64)
                    target_means = torch.from_numpy(c.reshape(-1, horizon).swapaxes(0, 1))
                else:
                    target_means = None
                reconcile_res = reconciler.perform_projection(
                    per_level_test_val_preds=per_level_test_val_preds,
                    aggregation_matrix=aggregation_matrix,
                    tags=tags,
                    target_means=target_means
                )
                y_rec[reconciler.get_fullname()] = reconcile_res.get_reconciled_means()
                rec_name_to_color[reconciler.get_fullname()] = reconciler.get_color()

            evaluation = evaluate(df=y_rec, tags=tags, metrics=metrics)
            melted_table = pd.melt(
                evaluation, id_vars=[ResultKey.LEVEL.value, ResultKey.METRIC.value], var_name=ResultKey.METHOD.value,
                value_name=ResultKey.VALUE.value
            )
            melted_table[ResultKey.DATASET.value] = hierarchical_dataset.name.value
            melted_table[ResultKey.HORIZON.value] = horizon
            melted_table[ResultKey.FOLD.value] = cross_val_ind
            melted_table[ResultKey.METHOD_COLOR.value] = melted_table[ResultKey.METHOD.value].map(
                lambda rec_name: rec_name_to_color[rec_name]
            )
            result_tables.append(melted_table)
        result_tables = pd.concat(result_tables, ignore_index=True)
        return result_tables

    @staticmethod
    def collect_results(
            dataset_names: list[HierarchicalDatasetNames], horizons: list[int], n_cross_vals: list[int] | int,
            metric_names: list[MetricName]
    ) -> pd.DataFrame:
        if isinstance(n_cross_vals, int):
            n_cross_vals = [n_cross_vals for _ in range(len(dataset_names))]
        result_tables = []
        for i, dataset_name in enumerate(dataset_names):
            result_table = ResultsCollector.collect_dataset_results(
                dataset_name=dataset_name, horizon=horizons[i], n_cross_val=n_cross_vals[i], metric_names=metric_names
            )
            result_tables.append(result_table)
        return pd.concat(result_tables, ignore_index=True)

    @staticmethod
    def collect_probabilistic_results(
            dataset_names: list[HierarchicalDatasetNames], horizons: list[int], n_cross_vals: list[int] | int,
    ) -> pd.DataFrame:
        if isinstance(n_cross_vals, int):
            n_cross_vals = [n_cross_vals for _ in range(len(dataset_names))]
        result_tables = []
        for i, dataset_name in enumerate(dataset_names):
            result_table = ResultsCollector.collect_probablistic_results(
                dataset_name=dataset_name, horizon=horizons[i], n_cross_val=n_cross_vals[i]
            )
            result_tables.append(result_table)
        return pd.concat(result_tables, ignore_index=True)

def collect_small_results() -> pd.DataFrame:
    dataset_names = [HierarchicalDatasetNames.TOURISM_SMALL, HierarchicalDatasetNames.WIKI2]
    horizons = [HierarchicalDataset.get_hierarchical_dataset(dataset_name=ds).default_horizon for ds in dataset_names]
    n_cv = 3
    return ResultsCollector.collect_results(
        dataset_names=dataset_names, horizons=horizons, n_cross_vals=n_cv,
        metric_names=[MetricName.RMSE, MetricName.MAE]
    )


def collect_tiny_results() -> pd.DataFrame:
    dataset_names = [HierarchicalDatasetNames.TOURISM_SMALL, HierarchicalDatasetNames.LABOUR]
    horizons = [HierarchicalDataset.get_hierarchical_dataset(dataset_name=ds).default_horizon for ds in dataset_names]
    n_cv = 2
    return ResultsCollector.collect_results(
        dataset_names=dataset_names, horizons=horizons, n_cross_vals=n_cv,
        metric_names=[MetricName.RMSE, MetricName.MAE]
    )


def collect_all_results() -> pd.DataFrame:
    dataset_names = [
        HierarchicalDatasetNames.TOURISM_SMALL,
        HierarchicalDatasetNames.LABOUR,
        HierarchicalDatasetNames.WIKI2,
        HierarchicalDatasetNames.TRAFFIC,
    ]
    n_cross_vals = 5
    horizons = [HierarchicalDataset.get_hierarchical_dataset(dataset_name=ds).default_horizon for ds in dataset_names]
    return ResultsCollector.collect_results(
        dataset_names=dataset_names, horizons=horizons, n_cross_vals=n_cross_vals,
        metric_names=[MetricName.RMSE, MetricName.MAE]
    )

def collect_all_probability_results() -> pd.DataFrame:
    dataset_names = [
        HierarchicalDatasetNames.TOURISM_SMALL,
        HierarchicalDatasetNames.LABOUR,
        HierarchicalDatasetNames.WIKI2,
        HierarchicalDatasetNames.TRAFFIC,
    ]
    n_cross_vals = 5
    horizons = [HierarchicalDataset.get_hierarchical_dataset(dataset_name=ds).default_horizon for ds in dataset_names]
    return ResultsCollector.collect_probabilistic_results(
        dataset_names=dataset_names, horizons=horizons, n_cross_vals=n_cross_vals,
    )
    
if __name__ == "__main__":
    results = collect_all_probability_results() 
    results.to_csv('results_prob.csv', index=False, header=True)
