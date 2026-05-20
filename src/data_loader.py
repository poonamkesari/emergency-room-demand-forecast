from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from .utils import ensure_directory

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
DEFAULT_HOURLY_PATH = DATA_DIR / "hourly_patient_volume.csv"
DEFAULT_REGULAR_PATH = DATA_DIR / "hourly_patient_volume_regular.csv"
DEFAULT_WEATHER_PATH = DATA_DIR / "weather_hourly_los_angeles.csv"
RAW_CSV_NAME = "Hospital ER_Data.csv"
RAW_PATH_CANDIDATES = [RAW_DIR / RAW_CSV_NAME, ROOT / "Notebook" / "data" / "raw" / RAW_CSV_NAME]

REQUIRED_RAW_COLUMNS = [
    "Patient Id",
    "Patient Admission Date",
    "Patient Waittime",
    "Patient Admission Flag",
]


def find_raw_csv() -> Path:
    for candidate in RAW_PATH_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Unable to locate raw ER source CSV. "
        f"Looked for: {', '.join(str(path) for path in RAW_PATH_CANDIDATES)}"
    )


def load_raw_data(raw_csv_path: Optional[Path] = None) -> pd.DataFrame:
    raw_csv_path = Path(raw_csv_path) if raw_csv_path is not None else find_raw_csv()
    return pd.read_csv(raw_csv_path)


def validate_raw_source(raw_er: pd.DataFrame) -> None:
    missing_columns = [column for column in REQUIRED_RAW_COLUMNS if column not in raw_er.columns]
    if missing_columns:
        raise ValueError(f"Missing required raw columns: {missing_columns}")

    parsed = pd.to_datetime(
        raw_er["Patient Admission Date"],
        format="%d-%m-%Y %H:%M",
        errors="coerce",
    )
    timestamp_failures = parsed.isna().sum()
    missing_patient_ids = raw_er["Patient Id"].isna().sum()
    negative_wait_times = (raw_er["Patient Waittime"] < 0).sum()
    admission_flag_valid = raw_er["Patient Admission Flag"].dropna().isin([True, False, 0, 1]).all()

    if timestamp_failures or missing_patient_ids or negative_wait_times or not admission_flag_valid:
        raise ValueError(
            "Raw source QA failed: "
            f"timestamp_failures={timestamp_failures}, "
            f"missing_patient_ids={missing_patient_ids}, "
            f"negative_wait_times={negative_wait_times}, "
            f"admission_flag_valid={admission_flag_valid}"
        )


def build_hourly_dataset(raw_er: pd.DataFrame, output_path: Optional[Path] = None) -> pd.DataFrame:
    validate_raw_source(raw_er)
    raw_er = raw_er.copy()
    raw_er["Patient_Admission_Timestamp"] = pd.to_datetime(
        raw_er["Patient Admission Date"],
        format="%d-%m-%Y %H:%M",
        errors="coerce",
    )
    missing_timestamps = raw_er["Patient_Admission_Timestamp"].isna().sum()
    if missing_timestamps:
        raise ValueError(f"Unable to parse {missing_timestamps} admission timestamps.")

    raw_er["DateTime_Hour"] = raw_er["Patient_Admission_Timestamp"].dt.floor("h")
    hourly_forecast = (
        raw_er.groupby("DateTime_Hour")
        .agg(
            Patient_Count=("Patient Id", "count"),
            Avg_Waittime=("Patient Waittime", "mean"),
            Admission_Count=("Patient Admission Flag", "sum"),
        )
        .reset_index()
    )
    hourly_forecast["Avg_Waittime"] = hourly_forecast["Avg_Waittime"].round(2)
    hourly_forecast["DateTime_Hour"] = hourly_forecast["DateTime_Hour"].dt.strftime("%Y-%m-%d %H:%M")

    if output_path is not None:
        output_path = Path(output_path)
        ensure_directory(output_path.parent)
        hourly_forecast.to_csv(output_path, index=False)
    return hourly_forecast


def load_hourly_data(hourly_path: Optional[Path] = None) -> pd.DataFrame:
    path = Path(hourly_path or DEFAULT_HOURLY_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Hourly source dataset not found at {path}")
    df = pd.read_csv(path)
    df["DateTime_Hour"] = pd.to_datetime(df["DateTime_Hour"])
    return df.sort_values("DateTime_Hour").reset_index(drop=True)


def regularize_hourly_data(hourly_df: pd.DataFrame, output_path: Optional[Path] = None) -> pd.DataFrame:
    hourly_df = hourly_df.copy()
    hourly_df["DateTime_Hour"] = pd.to_datetime(hourly_df["DateTime_Hour"])
    hourly_df = hourly_df.sort_values("DateTime_Hour").reset_index(drop=True)

    full_range = pd.date_range(
        hourly_df["DateTime_Hour"].min(),
        hourly_df["DateTime_Hour"].max(),
        freq="h",
    )
    regularized = pd.DataFrame({"DateTime_Hour": full_range}).merge(
        hourly_df,
        on="DateTime_Hour",
        how="left",
        indicator=True,
    )
    regularized["Was_Observed_Hour"] = regularized["_merge"].eq("both")
    regularized = regularized.drop(columns=["_merge"])
    regularized["Patient_Count"] = regularized["Patient_Count"].fillna(0).astype(int)
    regularized["Admission_Count"] = regularized["Admission_Count"].fillna(0).astype(int)

    if output_path is not None:
        output_path = Path(output_path)
        ensure_directory(output_path.parent)
        regularized.to_csv(output_path, index=False)

    return regularized
