import abc
from abc import abstractmethod
from enum import auto
from typing import ClassVar

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as f
from hierarchicalforecast.core import HierarchicalReconciliation
from hierarchicalforecast.methods import HReconciler, BottomUp, TopDown, ERM, MinTrace
from pydantic import BaseModel, ConfigDict

from src.datasets_manager import PerLevelTestValPreds
from src.utils.general_utils import ListableEnum


class ReconcilerName(ListableEnum):
    I_PROJ = auto()
    FULL_PROB_PAIRWISE = auto()
    MIN_T = auto()
    DO_NOTHING = auto()
    CLASSIC = auto()


class ReconciledRes(BaseModel, abc.ABC):
    model_config: ClassVar[ConfigDict] = {"arbitrary_types_allowed": True}

    @abstractmethod
    def get_reconciled_means(self) -> np.ndarray:
        pass


class VanillaReconciledRes(ReconciledRes):
    reconciled_means: np.ndarray
    rec_probs: torch.Tensor

    def get_reconciled_means(self) -> np.ndarray:
        return self.reconciled_means


class Reconciler(BaseModel, abc.ABC):
    name: ClassVar[ReconcilerName]

    @abstractmethod
    def get_fullname(self) -> str:
        pass

    @abstractmethod
    def perform_projection(
            self, per_level_test_val_preds: PerLevelTestValPreds, aggregation_matrix: pd.DataFrame,
            tags: dict[str, pd.DataFrame], **kwargs
    ) -> ReconciledRes:
        pass

    @abstractmethod
    def get_color(self) -> str:
        pass


class NoReconciler(Reconciler):
    name: ClassVar[str] = ReconcilerName.DO_NOTHING
    custom_name: str

    def perform_projection(
            self, per_level_test_val_preds: PerLevelTestValPreds, aggregation_matrix: pd.DataFrame,
            tags: dict[str, pd.DataFrame], **kwargs
    ) -> VanillaReconciledRes:
        probs = []
        for level in aggregation_matrix.index.unique():
            logits = per_level_test_val_preds.test_preds[level].pred_distribution.logits
            probs.append(f.softmax(logits))
        probs = torch.stack(probs).swapaxes(0, 1)
        return VanillaReconciledRes(reconciled_means=per_level_test_val_preds.flatten_test_pred_means,
        rec_probs=probs
        )

    def get_fullname(self) -> str:
        return self.custom_name

    def get_color(self) -> str:
        return "#ff7f0e"


class ClassicReconciler(Reconciler, abc.ABC):
    name: ClassVar[str] = ReconcilerName.CLASSIC

    @abstractmethod
    def get_classic_reconciler(self) -> HReconciler:
        pass

    def get_reconciler_basename(self) -> str:
        return str(self.get_classic_reconciler().__class__).split(".")[-1].split("'")[0]

    def perform_projection(
            self, per_level_test_val_preds: PerLevelTestValPreds, aggregation_matrix: pd.DataFrame,
            tags: dict[str, pd.DataFrame], **kwargs
    ) -> VanillaReconciledRes:
        reconciler = self.get_classic_reconciler()
        hrec = HierarchicalReconciliation(reconcilers=[reconciler])
        y_hat_df = per_level_test_val_preds.test_df.copy()
        y_hat_df["base_pred"] = per_level_test_val_preds.flatten_test_pred_means
        y_df = per_level_test_val_preds.val_test_df.copy()
        y_df["base_pred"] = per_level_test_val_preds.flatten_val_test_pred_means

        y_rec_df = hrec.reconcile(
            Y_hat_df=y_hat_df, Y_df=y_df, S=aggregation_matrix.reset_index(names='unique_id'), tags=tags
        )

        assert "base_pred/" in y_rec_df.columns[-1]
        assert len(y_rec_df.columns) == (len(y_df.columns) + 1)
        return VanillaReconciledRes(reconciled_means=y_rec_df.values[:, -1].flatten())


class BottomUpReconciler(ClassicReconciler):

    def get_fullname(self) -> str:
        return self.get_reconciler_basename()

    def get_color(self) -> str:
        return "#2ca02c"

    def get_classic_reconciler(self) -> HReconciler:
        return BottomUp()

class MinTMethodName(ListableEnum):
    def _generate_next_value_(name, start: ..., count: int, last_values: ...) -> str:
        """ Generate the next value when not given. """
        return name.lower()

    OLS = auto()
    MINT_SHRINK = auto()
    WLS_VAR  = auto()
    WLS_STRUCT = auto()


class MinTReconciler(ClassicReconciler):
    method: MinTMethodName
    nonnegative: bool

    def get_classic_reconciler(self) -> HReconciler:
        return MinTrace(method=self.method.value, nonnegative=self.nonnegative)

    def get_fullname(self) -> str:
        name = self.get_reconciler_basename() + f"-{self.method.value}"
        if self.nonnegative:
            name += "-nonneg"
        return name

    def get_color(self) -> str:
        match self.method:
            case MinTMethodName.OLS:
                return "#d62728"
            case MinTMethodName.MINT_SHRINK:
                return "#9467bd"
            case MinTMethodName.WLS_VAR:
                return "#8c564b"
            case MinTMethodName.WLS_STRUCT:
                return "#e377c2"
            case _:
                raise ValueError()


CLASSIC_RECONCILERS = [BottomUpReconciler()]
CLASSIC_RECONCILERS += [MinTReconciler(method=mint_method, nonnegative=True) for mint_method in MinTMethodName]
