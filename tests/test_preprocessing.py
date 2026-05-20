import numpy as np
import pandas as pd

from src.preprocessing import FEATURE_COLUMNS, WEATHER_COLUMNS, build_feature_table


def test_build_feature_table_creates_expected_columns() -> None:
    periods = 180
    dates = pd.date_range("2024-01-01", periods=periods, freq="h")
    regular_df = pd.DataFrame(
        {
            "DateTime_Hour": dates,
            "Patient_Count": np.arange(periods) % 12,
            "Avg_Waittime": 7.0,
            "Admission_Count": (np.arange(periods) % 2).astype(int),
            "Was_Observed_Hour": 1,
        }
    )
    weather_df = pd.DataFrame(
        {
            "DateTime_Hour": dates,
            "Temperature_F": 70.0,
            "Humidity_Pct": 55.0,
            "Rainfall_In": 0.0,
            "Temperature_Change_1h_F": 0.0,
            "Extreme_Weather_Flag": 0,
            "Weather_Code": 0,
        }
    )

    modeling_df = build_feature_table(regular_df, weather_df=weather_df)
    assert not modeling_df.empty
    for feature in FEATURE_COLUMNS:
        assert feature in modeling_df.columns
    assert modeling_df["Patient_Count"].dtype == int
    assert modeling_df["DateTime_Hour"].dtype == "datetime64[ns]"
    assert len(modeling_df) == periods - 168
