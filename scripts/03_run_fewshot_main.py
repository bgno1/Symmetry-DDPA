from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ddpa import MAIN_METHODS
from ddpa.train_eval import run_fewshot_main


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the main few-shot Macro-F1 comparison.")
    parser.add_argument("--task-plan", required=True)
    parser.add_argument("--malimg-train-dir", required=True)
    parser.add_argument("--malimg-val-dir", required=True)
    parser.add_argument("--malsim-checkpoint", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--shots", nargs="*", type=int, default=[1, 5, 10, 20])
    parser.add_argument("--methods", nargs="+", default=MAIN_METHODS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()

    run_fewshot_main(
        task_plan=args.task_plan,
        malimg_train_dir=args.malimg_train_dir,
        malimg_val_dir=args.malimg_val_dir,
        malsim_checkpoint=args.malsim_checkpoint,
        output_csv=args.output_csv,
        shots=args.shots,
        methods=args.methods,
        seed=args.seed,
        batch_size=args.batch_size,
        device_name=args.device,
        num_workers=args.num_workers,
        image_size=args.image_size,
        deterministic=args.deterministic,
    )


if __name__ == "__main__":
    main()
