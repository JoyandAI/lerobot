#!/usr/bin/env python

"""Demo: merge two LeRobot datasets into a single dataset.

Usage examples:
    python examples/dataset/merge_datasets_demo.py \
        --repo-id-a your_user/dataset_a \
        --repo-id-b your_user/dataset_b \
        --new-repo-id your_user/dataset_merged
"""

import argparse
from pathlib import Path

from lerobot.datasets.dataset_tools import merge_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import HF_LEROBOT_HOME


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge two LeRobot datasets.")
    parser.add_argument(
        "--repo-id-a",
        required=True,
        help="First source dataset repo id, e.g. 'your_user/dataset_a'.",
    )
    parser.add_argument(
        "--repo-id-b",
        required=True,
        help="Second source dataset repo id, e.g. 'your_user/dataset_b'.",
    )
    parser.add_argument(
        "--new-repo-id",
        required=True,
        help="Output dataset repo id, e.g. 'your_user/dataset_merged'.",
    )
    return parser.parse_args()


def _print_dataset(prefix: str, dataset: LeRobotDataset) -> None:
    print(
        f"[{prefix}] repo={dataset.repo_id}, root={dataset.root}, "
        f"episodes={dataset.meta.total_episodes}, frames={dataset.meta.total_frames}"
    )


def main() -> None:
    args = parse_args()
    output_dir = HF_LEROBOT_HOME / args.new_repo_id

    dataset_a = LeRobotDataset(args.repo_id_a)
    dataset_b = LeRobotDataset(args.repo_id_b)

    _print_dataset("source-a", dataset_a)
    _print_dataset("source-b", dataset_b)

    merged_dataset = merge_datasets(
        datasets=[dataset_a, dataset_b],
        output_repo_id=args.new_repo_id,
        output_dir=output_dir,
    )

    # Keep the top-level media directory layout aligned with the inputs.
    for dirname in ("images", "videos", "data", "meta"):
        if (Path(dataset_a.root) / dirname).exists() or (Path(dataset_b.root) / dirname).exists():
            (Path(merged_dataset.root) / dirname).mkdir(parents=True, exist_ok=True)

    _print_dataset("result", merged_dataset)


if __name__ == "__main__":
    main()
