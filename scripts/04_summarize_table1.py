from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from ddpa.metrics import summarize_mean_std
from ddpa.utils import ensure_dir, log_info


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize raw Table 1 few-shot results.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.input_csv)
    required = {"method", "K", "repeat", "macro_f1"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {sorted(missing)}")
    summary = summarize_mean_std(frame)
    output = Path(args.output_csv)
    ensure_dir(output.parent)
    summary.to_csv(output, index=False)
    log_info(f"Saved summary: {output}")


if __name__ == "__main__":
    main()
