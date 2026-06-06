from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ddpa.simclr import pretrain_malsim
from ddpa.utils import log_info, resolve_device, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain a Maldeb SimCLR ResNet-18 encoder.")
    parser.add_argument("--maldeb-dir", required=True)
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed, deterministic=args.deterministic)
    info = pretrain_malsim(
        maldeb_dir=args.maldeb_dir,
        output_checkpoint=args.output_checkpoint,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        temperature=args.temperature,
        image_size=args.image_size,
        seed=args.seed,
        device=resolve_device(args.device),
        num_workers=args.num_workers,
    )
    log_info(f"Saved checkpoint: {info['checkpoint_path']}")
    log_info(f"Checkpoint SHA256: {info['checkpoint_sha256']}")


if __name__ == "__main__":
    main()
