import numpy as np
import pandas as pd

from src.train import create_supervised_targets, encode_features, split_chronological


def test_create_supervised_targets_shifts_labels() -> None:
    df = pd.DataFrame(
        {
            "DateTime_Hour": pd.date_range("2024-01-01", periods=10, freq="h"),
            "Patient_Count": np.arange(10),
        }
    )
    supervised_df, target_columns = create_supervised_targets(df)
    assert target_columns == ["Target_1h", "Target_6h", "Target_24h", "Target_72h"]
    assert supervised_df["Target_1h"].iloc[0] == 1
    assert supervised_df["Target_6h"].iloc[0] == 6


def test_encode_features_and_split_chronological() -> None:
    df = pd.DataFrame(
        {
            "DateTime_Hour": pd.date_range("2024-01-01", periods=20, freq="h"),
            "Patient_Count": np.arange(20),
            "Lag_1h": np.arange(20),
            "Hour": np.arange(20) % 24,
            "Target_1h": np.arange(20) + 1,
            "Target_6h": np.arange(20) + 6,
            "Target_24h": np.arange(20) + 24,
            "Target_72h": np.arange(20) + 72,
        }
    )
    X, y, origins = encode_features(df, ["Lag_1h", "Hour"], ["Target_1h", "Target_6h", "Target_24h", "Target_72h"])
    splits = split_chronological(X, y, origins, train_ratio=0.6, validation_ratio=0.2)
    assert len(splits["X_train"]) == 12
    assert len(splits["X_val"]) == 4
    assert len(splits["X_test"]) == 4
    assert splits["date_train"].iloc[0] == pd.Timestamp("2024-01-01 00:00:00")
