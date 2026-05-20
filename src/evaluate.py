from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .utils import grouped_error_summary, regression_metrics


def build_metrics_frame(
    prediction_store: dict[str, pd.DataFrame],
    y_test: pd.DataFrame,
    model_group_map: dict[str, str],
    forecast_horizons: dict[int, str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_name, predictions in prediction_store.items():
        model_group = model_group_map.get(model_name, "Model")
        for horizon, target_column in forecast_horizons.items():
            predicted = predictions[target_column].astype(float)
            actual = y_test[target_column].astype(float)
            rows.append(
                {
                    "Model": model_name,
                    "Model_Group": model_group,
                    "Horizon": f"{horizon}h",
                    **regression_metrics(actual, predicted),
                }
            )
    return pd.DataFrame(rows)


def compute_operational_metrics(
    best_model_by_horizon: pd.DataFrame,
    prediction_store: dict[str, pd.DataFrame],
    y_test: pd.DataFrame,
    forecast_origins: pd.Series,
    forecast_horizons: dict[int, str],
    peak_quantile: float = 0.75,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for horizon, target_column in forecast_horizons.items():
        horizon_label = f"{horizon}h"
        row = best_model_by_horizon.loc[best_model_by_horizon["Horizon"] == horizon_label]
        if row.empty:
            continue
        model_name = row["Model"].iloc[0]
        predictions = prediction_store[model_name][target_column].astype(float).clip(lower=0.0)
        actual = y_test[target_column].astype(float)
        error_df = pd.DataFrame(
            {
                "DateTime_Hour": pd.to_datetime(forecast_origins.values),
                "Actual": actual.values,
                "Predicted": predictions.values,
            },
            index=y_test.index,
        )
        error_df["Residual"] = error_df["Actual"] - error_df["Predicted"]
        error_df["Hour"] = error_df["DateTime_Hour"].dt.hour
        error_df["Day_Of_Week"] = error_df["DateTime_Hour"].dt.day_name()

        peak_threshold = error_df["Actual"].quantile(peak_quantile)
        peak_df = error_df[error_df["Actual"] >= peak_threshold]

        rows.append(
            {
                "Horizon": horizon_label,
                "Best_Model": model_name,
                "Peak_Threshold": float(peak_threshold),
                "Peak_Rows": int(len(peak_df)),
                "Peak_RMSE": float(np.sqrt(np.mean(np.square(peak_df["Residual"])))),
                "Peak_MAE": float(np.mean(np.abs(peak_df["Residual"]))),
                "Peak_Underprediction_Rate": float(np.mean(peak_df["Residual"] > 0)),
                "Worst_Hour_By_MAE": int(grouped_error_summary(error_df, "Hour").sort_values("MAE", ascending=False).iloc[0]["Hour"]),
                "Worst_Day_By_MAE": str(grouped_error_summary(error_df, "Day_Of_Week").sort_values("MAE", ascending=False).iloc[0]["Day_Of_Week"]),
            }
        )
    return pd.DataFrame(rows)
