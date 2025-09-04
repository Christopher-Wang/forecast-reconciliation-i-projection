import argparse

import pyrootutils

root = pyrootutils.setup_root(search_from=__file__, indicator="pyproject.toml", pythonpath=True, cwd=True)

from src.datasets_manager import HierarchicalDatasetNames, HierarchicalDataset
from src.utils.predictions_utils import DeviceLikeType


def main(dataset_name: str, device: DeviceLikeType) -> None:
    hierarchical_dataset = HierarchicalDataset.get_hierarchical_dataset(
        dataset_name=HierarchicalDatasetNames.get_enum_element(value=dataset_name)
    )

    hierarchical_dataset.get_cross_val_data(n_cross_val=5, horizon=None, device=device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run predictions.")
    parser.add_argument(
        "--dataset", type=str, nargs="+", choices=[e.value for e in HierarchicalDatasetNames], required=True,
        help="Dataset names"
    )
    parser.add_argument("--device", type=str, required=True, help="Device to use")
    args = parser.parse_args()

    for dataset_name_ in args.dataset:
        main(dataset_name=dataset_name_, device=args.device)


# [fr1] taskset -c 0-14 python scripts/pred_generation.py --dataset TourismSmall --device 'cuda:0'
# [fr2] taskset -c 15-29 python scripts/pred_generation.py --dataset Labour --device 'cuda:1'
# [fr3] taskset -c 30-44 python scripts/pred_generation.py --dataset Wiki2 --device 'cuda:2'
# [fr4] taskset -c 45-59 python scripts/pred_generation.py --dataset Traffic --device 'cuda:3'
# [fr5] taskset -c 60-74 python scripts/pred_generation.py --dataset TourismLarge --device 'cuda:0'
