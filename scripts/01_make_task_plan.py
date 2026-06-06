from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ddpa.datasets import build_fixed_task_plan, load_malimg_splits, save_task_plan
from ddpa.utils import log_info, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Create fixed Malimg support/query tasks for the main few-shot comparison.")
    parser.add_argument("--malimg-train-dir", required=True)
    parser.add_argument("--malimg-val-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--shots", nargs="+", type=int, default=[1, 5, 10, 20])
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed, deterministic=args.deterministic)
    train_ds, val_ds, _ = load_malimg_splits(args.malimg_train_dir, args.malimg_val_dir, image_size=args.image_size)
    plan = build_fixed_task_plan(
        train_ds=train_ds,
        query_ds=val_ds,
        train_dir=args.malimg_train_dir,
        val_dir=args.malimg_val_dir,
        shots=args.shots,
        repeats=args.repeats,
        seed=args.seed,
    )
    save_task_plan(plan, args.output_json)
    log_info(f"Saved task plan: {args.output_json}")


if __name__ == "__main__":
    main()
