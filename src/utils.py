from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_STATE = 42


def set_random_seed(seed: int = RANDOM_STATE) -> None:
    np.random.seed(seed)


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def clip_nonnegative(predictions: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(predictions, dtype=float), 0.0, None)


def zero_safe_mape(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    nonzero_mask = y_true_arr != 0
    if not nonzero_mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true_arr[nonzero_mask] - y_pred_arr[nonzero_mask]) / y_true_arr[nonzero_mask])) * 100)


def regression_metrics(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> dict[str, float]:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    residuals = y_true_arr - y_pred_arr
    return {
        "RMSE": float(np.sqrt(np.mean(np.square(residuals)))),
        "MAE": float(np.mean(np.abs(residuals))),
        "MAPE": zero_safe_mape(y_true_arr, y_pred_arr),
        "Mean_Bias": float(np.mean(residuals)),
        "Underprediction_Rate": float(np.mean(residuals > 0)),
    }


def grouped_error_summary(df: pd.DataFrame, group_column: str) -> pd.DataFrame:
    rows = []
    for group_value, group_df in df.groupby(group_column):
        rows.append(
            {
                group_column: group_value,
                "Rows": len(group_df),
                "MAE": float(np.mean(np.abs(group_df["Actual"] - group_df["Predicted"]))),
                "RMSE": float(np.sqrt(np.mean(np.square(group_df["Actual"] - group_df["Predicted"])))),
                "Mean_Bias": float(np.mean(group_df["Actual"] - group_df["Predicted"])),
                "Underprediction_Rate": float(np.mean((group_df["Actual"] - group_df["Predicted"]) > 0)),
            }
        )
    return pd.DataFrame(rows)
