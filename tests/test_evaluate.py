import pandas as pd

from src.evaluate import build_metrics_frame, compute_operational_metrics


def test_build_metrics_frame_returns_expected_columns() -> None:
    y_test = pd.DataFrame(
        {
            "Target_1h": [10.0, 12.0],
            "Target_6h": [15.0, 16.0],
            "Target_24h": [20.0, 18.0],
            "Target_72h": [25.0, 27.0],
        }
    )
    predictions = {
        "Persistence": pd.DataFrame(
            {
                "Target_1h": [10.0, 11.0],
                "Target_6h": [14.0, 16.0],
                "Target_24h": [21.0, 18.0],
                "Target_72h": [24.0, 28.0],
            }
        )
    }
    metrics = build_metrics_frame(predictions, y_test, {"Persistence": "Simple Benchmark"}, {1: "Target_1h", 6: "Target_6h", 24: "Target_24h", 72: "Target_72h"})
    assert "RMSE" in metrics.columns
    assert metrics.loc[metrics["Horizon"] == "1h", "Model"].iloc[0] == "Persistence"


def test_compute_operational_metrics_creates_summary() -> None:
    y_test = pd.DataFrame(
        {
            "Target_1h": [10.0, 12.0, 11.0, 14.0],
            "Target_6h": [20.0, 19.0, 21.0, 22.0],
            "Target_24h": [30.0, 31.0, 29.0, 28.0],
            "Target_72h": [35.0, 36.0, 34.0, 33.0],
        }
    )
    predictions = {
        "BestModel": pd.DataFrame(
            {
                "Target_1h": [9.0, 13.0, 11.0, 15.0],
                "Target_6h": [19.0, 18.0, 22.0, 23.0],
                "Target_24h": [31.0, 30.0, 28.0, 27.0],
                "Target_72h": [34.0, 37.0, 33.0, 32.0],
            }
        )
    }
    forecast_origins = pd.date_range("2024-01-01", periods=4, freq="h")
    best_model_by_horizon = pd.DataFrame(
        {
            "Model": ["BestModel"] * 4,
            "Horizon": ["1h", "6h", "24h", "72h"],
        }
    )
    summary = compute_operational_metrics(
        best_model_by_horizon,
        predictions,
        y_test,
        forecast_origins,
        {1: "Target_1h", 6: "Target_6h", 24: "Target_24h", 72: "Target_72h"},
    )
    assert not summary.empty
    assert set(summary["Horizon"]) == {"1h", "6h", "24h", "72h"}
