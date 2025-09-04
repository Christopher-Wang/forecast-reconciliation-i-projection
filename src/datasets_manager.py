from __future__ import annotations

from enum import auto
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame
from datasetsforecast.hierarchical import HierarchicalData
from pydantic import BaseModel, ConfigDict
from tabpfn import TabPFNRegressor
from tabpfn.architectures.base.bar_distribution import FullSupportBarDistribution
from tabpfn_time_series import FeatureTransformer
from tabpfn_time_series.features import RunningIndexFeature, CalendarFeature, AutoSeasonalFeature
from tqdm import tqdm

from src.utils.general_utils import get_project_root, np_date_to_str, get_cross_val_dates, ListableEnum
from src.utils.predictions_utils import RiemannDistrib, PFNPreds, DeviceLikeType


class PredictorModelName(ListableEnum):
    TAB_PFN_TS = auto()


class HierarchicalDatasetNames(ListableEnum):
    def _generate_next_value_(name, start: ..., count: int, last_values: ...) -> str:
        """Generate the next value when not given."""
        return name.title().replace("_", "")

    TOURISM_SMALL = auto()
    TOURISM_LARGE = auto()
    WIKI2 = auto()
    LABOUR = auto()
    TRAFFIC = auto()


class HierarchicalPredsConfig(BaseModel):
    dataset_name: HierarchicalDatasetNames
    level: str
    context_cutoff_date: np.datetime64
    test_cutoff_date: np.datetime64 | None
    predictor_model_name: PredictorModelName

    model_config = {
        "arbitrary_types_allowed": True
    }

    def get_save_path(self) -> Path:
        save_path = Path("/nfs/rlteam3/forecast_reconciliation/pred_data")
        save_path /= self.dataset_name.value.replace(" ", "__")
        save_path /= str(self.level).replace(" ", "__")
        date_id = np_date_to_str(date=self.context_cutoff_date) + "__" + np_date_to_str(date=self.test_cutoff_date)
        save_path /= date_id
        save_path /= self.predictor_model_name.value.replace(" ", "__")

        return save_path


def generate_tsdf(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_tsdf = TimeSeriesDataFrame.from_data_frame(
        train_df.rename(columns={'y': 'target'}),
        timestamp_column='ds',
        id_column='unique_id'
    )

    _test_df = test_df.rename(columns={'y': 'target'})
    _test_df['target'] = np.nan
    test_tsdf = TimeSeriesDataFrame.from_data_frame(
        _test_df,
        timestamp_column='ds',
        id_column='unique_id'
    )

    selected_features = [
        RunningIndexFeature(),
        CalendarFeature(),
        AutoSeasonalFeature(),
    ]

    feature_transformer = FeatureTransformer(selected_features)
    train_tsdf, test_tsdf = feature_transformer.transform(train_tsdf, test_tsdf)
    return train_tsdf, test_tsdf


class PerLevelTestValPreds(BaseModel):
    test_df: pd.DataFrame
    val_test_df: pd.DataFrame
    test_preds: dict[str, PFNPreds]
    val_test_preds: dict[str, PFNPreds]
    flatten_test_pred_means: np.ndarray
    flatten_val_test_pred_means: np.ndarray

    model_config: ClassVar[ConfigDict] = {"arbitrary_types_allowed": True}


class HierarchicalDataset(BaseModel):
    name: ClassVar[HierarchicalDatasetNames]
    default_horizon: ClassVar[int]

    def get_data(self) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
        y_df, aggregation_matrix, tags = HierarchicalData.load('./data', self.name.value)
        y_df['ds'] = pd.to_datetime(y_df['ds'])
        return y_df, aggregation_matrix, tags

    @staticmethod
    def get_hierarchical_dataset(dataset_name: HierarchicalDatasetNames) -> HierarchicalDataset:
        match dataset_name:
            case HierarchicalDatasetNames.TOURISM_SMALL:
                return TourismSmall()
            case HierarchicalDatasetNames.TOURISM_LARGE:
                return TourismLarge()
            case HierarchicalDatasetNames.WIKI2:
                return Wiki2()
            case HierarchicalDatasetNames.LABOUR:
                return Labour()
            case HierarchicalDatasetNames.TRAFFIC:
                return Traffic()
            case _:
                raise ValueError()

    def get_cross_val_data(self, n_cross_val: int, horizon: int | None, device: DeviceLikeType) -> tuple[
        list[PerLevelTestValPreds], pd.DataFrame, dict[str, np.ndarray]]:
        y_df, aggregation_matrix, tags = self.get_data()
        dates: np.ndarray[np.datetime64] = np.sort(y_df["ds"].unique())
        if horizon is None:
            horizon = self.default_horizon

        val_ctx_cuts, train_cuts, test_cuts = get_cross_val_dates(dates=dates, n_cross_val=n_cross_val, horizon=horizon)
        predictions = []
        for i in range(n_cross_val):
            val_ctx_cut, train_cut, test_cut = val_ctx_cuts[i], train_cuts[i], test_cuts[i]

            train_df, test_df = y_df[y_df['ds'] < train_cut[1]], y_df[y_df['ds'] >= train_cut[1]]
            if test_cut[1] is not None:
                test_df = test_df[test_df['ds'] < test_cut[1]]

            val_ctx_df, val_test_df = train_df[train_df['ds'] < val_ctx_cut[1]], train_df[
                (train_df['ds'] >= val_ctx_cut[1])]

            train_tsdf, test_tsdf = generate_tsdf(train_df=train_df, test_df=test_df)
            val_ctx_tsdf, val_test_tsdf = generate_tsdf(train_df=val_ctx_df, test_df=val_test_df)
            test_preds, flatten_test_pred_means = self.get_preds(
                ctx_tsdf=train_tsdf, test_tsdf=test_tsdf, context_cutoff_date=train_cut[1],
                test_cutoff_date=test_cut[1], aggregation_matrix=aggregation_matrix, device=device,
                tqdm_desc=f"{self.name.value} - Fold {i} - Train/Test"
            )
            val_test_preds, flatten_val_test_pred_means = self.get_preds(
                ctx_tsdf=val_ctx_tsdf, test_tsdf=val_test_tsdf, context_cutoff_date=val_ctx_cut[1],
                test_cutoff_date=train_cut[1], aggregation_matrix=aggregation_matrix, device=device,
                tqdm_desc=f"{self.name.value} - Fold {i} - Validation"
            )
            eval_ready_data = PerLevelTestValPreds(
                test_df=test_df, val_test_df=val_test_df, test_preds=test_preds, val_test_preds=val_test_preds,
                flatten_test_pred_means=flatten_test_pred_means, flatten_val_test_pred_means=flatten_val_test_pred_means
            )

            predictions.append(eval_ready_data)
        return predictions, aggregation_matrix, tags

    def get_preds(
            self, ctx_tsdf: pd.DataFrame, test_tsdf: pd.DataFrame, context_cutoff_date: np.datetime64,
            test_cutoff_date: np.datetime64, aggregation_matrix: pd.DataFrame, device: DeviceLikeType,
            tqdm_desc: str
    ) -> tuple[dict[str, PFNPreds], np.ndarray]:
        """

        Args:
            ctx_tsdf:
            test_tsdf:
            context_cutoff_date:
            test_cutoff_date:
            aggregation_matrix:
            device:
            tqdm_desc:

        Returns:
            per_level_preds: prediction per level
            flatten_pred_mean: flatten mean prediction (one row per level)
        """
        reg = TabPFNRegressor(random_state=42, device=device)
        pred_model_name = PredictorModelName.TAB_PFN_TS
        preds: dict[str, PFNPreds] = {}

        per_level_pred_mean: list[np.ndarray] = []

        for level in tqdm(aggregation_matrix.index.unique(), desc=tqdm_desc):
            pred_name = " * ".join([self.name.value, level, pred_model_name.value])
            config = HierarchicalPredsConfig(
                dataset_name=self.name, level=level, context_cutoff_date=context_cutoff_date,
                test_cutoff_date=test_cutoff_date, predictor_model_name=pred_model_name
            )
            pred_save_path = config.get_save_path()

            pfn_pred = PFNPreds.load(dirpath=pred_save_path, fault_tolerant=True)
            if pfn_pred is None:
                x_train = ctx_tsdf.drop(columns='target')
                y_train = ctx_tsdf['target']
                x_test = test_tsdf.drop(columns='target')

                reg.fit(x_train.loc[level], y_train.loc[level])
                y_pred = reg.predict(x_test.loc[level], output_type='full')
                pred_distribution = RiemannDistrib(
                    full_support_bar=FullSupportBarDistribution(borders=y_pred['criterion'].borders),
                    logits=y_pred['logits']
                )
                pfn_pred = PFNPreds(
                    name=pred_name, pred_distribution=pred_distribution, train_mean=reg.y_train_mean_,
                    train_std=reg.y_train_std_
                )

                if y_pred['criterion'].borders[0] != (reg.bardist_.borders * reg.y_train_std_ + reg.y_train_mean_)[0]:
                    raise RuntimeError()

                assert np.allclose(pfn_pred.mean_pred.detach().cpu().numpy(), y_pred['mean'], rtol=1e-5)
                pfn_pred.save(dirpath=pred_save_path)

            preds[level] = pfn_pred
            per_level_pred_mean.append(pfn_pred.mean_pred.detach().cpu().numpy())

        return preds, np.array(per_level_pred_mean).flatten()


class TourismSmall(HierarchicalDataset):
    name: ClassVar[HierarchicalDatasetNames] = HierarchicalDatasetNames.TOURISM_SMALL
    default_horizon: ClassVar[int] = 4


class TourismLarge(HierarchicalDataset):
    name: ClassVar[HierarchicalDatasetNames] = HierarchicalDatasetNames.TOURISM_LARGE
    default_horizon: ClassVar[int] = 12  # https://proceedings.mlr.press/v139/rangapuram21a/rangapuram21a.pdf


class Wiki2(HierarchicalDataset):
    name: ClassVar[HierarchicalDatasetNames] = HierarchicalDatasetNames.WIKI2
    default_horizon: ClassVar[int] = 7  # https://arxiv.org/pdf/2204.10414


class Labour(HierarchicalDataset):
    name: ClassVar[HierarchicalDatasetNames] = HierarchicalDatasetNames.LABOUR
    default_horizon: ClassVar[int] = 8  # (https://arxiv.org/pdf/2204.10414, Learning Opt Proj for Forecast Rec)


class Traffic(HierarchicalDataset):
    name: ClassVar[HierarchicalDatasetNames] = HierarchicalDatasetNames.TRAFFIC
    default_horizon: ClassVar[int] = 7  # https://arxiv.org/pdf/2204.10414
