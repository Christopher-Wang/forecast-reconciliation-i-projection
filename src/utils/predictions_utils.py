from __future__ import annotations

import json
from pathlib import Path

import torch
from pydantic import BaseModel
from tabpfn.architectures.base.bar_distribution import FullSupportBarDistribution
from typing_extensions import TypeAlias

DeviceLikeType: TypeAlias = str | torch.device | int


class RiemannDistrib(BaseModel):
    full_support_bar: FullSupportBarDistribution
    logits: torch.Tensor  # shape (batch, nbins)

    model_config = {
        "arbitrary_types_allowed": True
    }

    def to(self, dtype: torch.dtype | None = None, device: DeviceLikeType | None = None) -> RiemannDistrib:
        return RiemannDistrib(
            full_support_bar=self.full_support_bar.to(dtype=dtype, device=device),
            logits=self.logits.to(dtype=dtype, device=device)
        )

    @property
    def bucket_widths(self) -> torch.Tensor:
        return self.full_support_bar.bucket_widths

    @property
    def bucket_means(self) -> torch.Tensor:
        bucket_means = self.full_support_bar.borders[:-1] + self.full_support_bar.bucket_widths / 2
        side_normals = (
            FullSupportBarDistribution.halfnormal_with_p_weight_before(self.full_support_bar.bucket_widths[0]),
            FullSupportBarDistribution.halfnormal_with_p_weight_before(self.full_support_bar.bucket_widths[-1]),
        )
        bucket_means[0] = -side_normals[0].mean + self.full_support_bar.borders[1]
        bucket_means[-1] = side_normals[1].mean + self.full_support_bar.borders[-2]
        return bucket_means

    @property
    def mean(self) -> torch.Tensor:
        """
         Average prediction value for each timestep

        Returns:
            mean: tensor of shape (batch, 1)
        """
        return self.full_support_bar.mean(logits=self.logits)

    def requires_grad_(self, mode: bool) -> RiemannDistrib:
        self.logits.requires_grad_(mode=mode)
        return self

    def save(self, path: Path) -> None:
        torch.save(
            {
                "full_support_bar": self.full_support_bar,
                "logits": self.logits,
            }, path
        )

    @classmethod
    def load(cls, path: Path) -> RiemannDistrib:
        with torch.serialization.safe_globals([FullSupportBarDistribution]):
            data = torch.load(path)
            return cls(full_support_bar=data["full_support_bar"], logits=data["logits"])


class PFNPreds(BaseModel):
    name: str
    pred_distribution: RiemannDistrib
    train_mean: float
    train_std: float

    model_config = {
        "arbitrary_types_allowed": True
    }

    def to(self, dtype: torch.dtype | None = None, device: DeviceLikeType | None = None) -> PFNPreds:
        return PFNPreds(
            name=self.name,
            pred_distribution=self.pred_distribution.to(dtype=dtype, device=device),
            train_mean=self.train_mean,
            train_std=self.train_std
        )

    def requires_grad_(self, mode: bool) -> None:
        self.pred_distribution.requires_grad_(mode=mode)

    @property
    def mean_pred(self) -> torch.Tensor:
        """
         Average prediction value for each timestep

        Returns:
            mean: tensor of shape (batch, 1)
        """
        return self.pred_distribution.mean

    def save(self, dirpath: Path) -> None:
        """
        Save PFNPreds object to a directory:
        - metadata.json: stores name, train_mean, train_std
        - distrib.pt: stores pred_distribution tensors
        """
        dirpath.mkdir(parents=True, exist_ok=True)

        # Save metadata
        metadata = {
            "name": self.name,
            "train_mean": self.train_mean,
            "train_std": self.train_std,
        }
        with open(dirpath / "metadata.json", "w") as f:
            json.dump(metadata, f)

        # Save distribution
        self.pred_distribution.save(dirpath / "distrib.pt")
        print(f"Saved distribution in {dirpath}")

    @classmethod
    def load(cls, dirpath: Path, fault_tolerant: bool) -> PFNPreds | None:
        """
        Load PFNPreds object from a directory.

        Args:
            dirpath: folder where the object should be saved
            fault_tolerant: whether to just return None if the object is not found

        """
        # Load metadata
        metadata_path = dirpath / "metadata.json"
        if not metadata_path.exists():
            if fault_tolerant:
                return None
            raise FileNotFoundError(metadata_path)

        with open(dirpath / "metadata.json", "r") as f:
            metadata = json.load(f)

        # Load distribution
        distrib = RiemannDistrib.load(path=dirpath / "distrib.pt")

        return cls(
            name=metadata["name"],
            pred_distribution=distrib,
            train_mean=metadata["train_mean"],
            train_std=metadata["train_std"],
        )
