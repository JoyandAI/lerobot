#!/usr/bin/env python

"""Demo: delete one or more episodes from a LeRobot dataset.

Usage examples:
    python examples/dataset/delete_episode_demo.py \
        --repo-id your_user/your_dataset \
        --delete 3

    python examples/dataset/delete_episode_demo.py \
        --repo-id lerobot/pusht \
        --delete 0 2 \
        --new-repo-id lerobot/pusht_filtered
"""

import argparse
from pathlib import Path

from lerobot.datasets.dataset_tools import delete_episodes
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import HF_LEROBOT_HOME


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Delete episodes from a LeRobot dataset.")
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Source dataset repo id, e.g. 'your_user/your_dataset'.",
    )
    parser.add_argument(
        "--delete",
        nargs="+",
        required=True,
        type=int,
        help="Episode indices to delete, e.g. --delete 3 or --delete 3 7 9.",
    )
    parser.add_argument(
        "--new-repo-id",
        default=None,
        help="Output dataset repo id. Default: <repo-id>_modified.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_repo_id = args.new_repo_id or f"{args.repo_id}_modified"

    dataset = LeRobotDataset(args.repo_id)
    print(
        f"[source] repo={args.repo_id}, episodes={dataset.meta.total_episodes}, "
        f"frames={dataset.meta.total_frames}"
    )

    output_dir = HF_LEROBOT_HOME / target_repo_id

    new_dataset = delete_episodes(
        dataset=dataset,
        episode_indices=args.delete,
        output_dir=output_dir,
        repo_id=target_repo_id,
    )

    # Keep the top-level media directory layout aligned with the source dataset.
    # Some downstream tools expect these directories to exist even for video datasets.
    for dirname in ("images", "videos", "data", "meta"):
        src_dir = Path(dataset.root) / dirname
        if src_dir.exists():
            (Path(new_dataset.root) / dirname).mkdir(parents=True, exist_ok=True)

    print(
        f"[result] repo={new_dataset.repo_id}, root={new_dataset.root}, "
        f"episodes={new_dataset.meta.total_episodes}, frames={new_dataset.meta.total_frames}"
    )


if __name__ == "__main__":
    main()
