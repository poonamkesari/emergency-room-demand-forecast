import pandas as pd
import pytest

from src.data_loader import build_hourly_dataset, validate_raw_source


def test_build_hourly_dataset_aggregates_by_hour() -> None:
    raw = pd.DataFrame(
        {
            "Patient Id": [1, 2, 3],
            "Patient Admission Date": ["01-01-2024 08:15", "01-01-2024 08:45", "01-01-2024 09:05"],
            "Patient Waittime": [5.0, 10.0, 12.0],
            "Patient Admission Flag": [1, 0, 1],
        }
    )
    hourly = build_hourly_dataset(raw)
    assert len(hourly) == 2
    assert hourly.loc[hourly["DateTime_Hour"] == "2024-01-01 08:00", "Patient_Count"].iloc[0] == 2
    assert hourly.loc[hourly["DateTime_Hour"] == "2024-01-01 09:00", "Patient_Count"].iloc[0] == 1


def test_validate_raw_source_raises_missing_column() -> None:
    raw = pd.DataFrame(
        {
            "Patient Id": [1],
            "Patient Admission Date": ["01-01-2024 08:15"],
            "Patient Waittime": [5.0],
        }
    )
    with pytest.raises(ValueError, match="Missing required raw columns"):
        validate_raw_source(raw)
