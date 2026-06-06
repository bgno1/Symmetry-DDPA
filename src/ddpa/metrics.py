from __future__ import annotations

from collections.abc import Iterable

import pandas as pd
from sklearn.metrics import f1_score


def macro_f1(y_true: Iterable[int], y_pred: Iterable[int]) -> float:
    return float(f1_score(list(y_true), list(y_pred), average="macro", zero_division=0))


def summarize_mean_std(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method, shot), group in frame.groupby(["method", "K"], sort=False):
        rows.append(
            {
                "method": method,
                "K": int(shot),
                "mean_macro_f1": float(group["macro_f1"].mean()),
                "std_macro_f1": float(group["macro_f1"].std(ddof=1)) if len(group) > 1 else 0.0,
                "n_repeats": int(group["macro_f1"].count()),
            }
        )
    return pd.DataFrame(rows)
