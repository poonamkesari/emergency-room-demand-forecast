from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from .data_loader import DEFAULT_WEATHER_PATH
from .utils import ensure_directory

WEATHER_COLUMNS = [
    "Temperature_F",
    "Temperature_Change_1h_F",
    "Humidity_Pct",
    "Rainfall_In",
    "Extreme_Weather_Flag",
]

CALENDAR_FEATURE_COLUMNS = [
    "Hour",
    "Day_Of_Week",
    "Is_Weekend",
    "Month",
    "Holiday_Flag",
    "Day_After_Holiday",
    "Was_Observed_Hour",
]

LAG_HOURS = [1, 2, 3, 6, 12, 24, 48, 72, 168]
LAG_COLUMNS = [f"Lag_{lag}h" for lag in LAG_HOURS]

ROLLING_MEAN_WINDOWS = [3, 6, 24, 168]
ROLLING_STD_WINDOWS = [24, 168]
ROLLING_FEATURE_COLUMNS = [f"Rolling_Mean_{window}h" for window in ROLLING_MEAN_WINDOWS] + [
    f"Rolling_Std_{window}h" for window in ROLLING_STD_WINDOWS
]

CYCLICAL_FEATURE_COLUMNS = [
    "Hour_Sin",
    "Hour_Cos",
    "Day_Of_Week_Sin",
    "Day_Of_Week_Cos",
    "Month_Sin",
    "Month_Cos",
]

FEATURE_COLUMNS = (
    CALENDAR_FEATURE_COLUMNS
    + ["Is_Flu_Season"]
    + WEATHER_COLUMNS
    + LAG_COLUMNS
    + ROLLING_FEATURE_COLUMNS
    + CYCLICAL_FEATURE_COLUMNS
)

FORECAST_HORIZONS = {1: "Target_1h", 6: "Target_6h", 24: "Target_24h", 72: "Target_72h"}


def load_or_fetch_los_angeles_weather(
    start_timestamp: pd.Timestamp,
    end_timestamp: pd.Timestamp,
    cache_path: Path | str | None = None,
) -> pd.DataFrame:
    cache_path = Path(cache_path or DEFAULT_WEATHER_PATH)
    start_date = pd.Timestamp(start_timestamp).date().isoformat()
    end_date = pd.Timestamp(end_timestamp).date().isoformat()

    if cache_path.exists():
        weather_df = pd.read_csv(cache_path, parse_dates=["DateTime_Hour"])
        if weather_df["DateTime_Hour"].min() <= pd.Timestamp(start_timestamp) and weather_df["DateTime_Hour"].max() >= pd.Timestamp(end_timestamp):
            return weather_df

    params = {
        "latitude": 34.0522,
        "longitude": -118.2437,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,relative_humidity_2m,rain,weather_code",
        "temperature_unit": "fahrenheit",
        "precipitation_unit": "inch",
        "timezone": "America/Los_Angeles",
    }
    url = "https://archive-api.open-meteo.com/v1/archive?" + urlencode(params)
    with urlopen(url, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))

    hourly_weather = payload["hourly"]
    weather_df = pd.DataFrame(
        {
            "DateTime_Hour": pd.to_datetime(hourly_weather["time"]),
            "Temperature_F": hourly_weather["temperature_2m"],
            "Humidity_Pct": hourly_weather["relative_humidity_2m"],
            "Rainfall_In": hourly_weather["rain"],
            "Weather_Code": hourly_weather["weather_code"],
        }
    ).sort_values("DateTime_Hour")
    weather_df["Temperature_Change_1h_F"] = weather_df["Temperature_F"].diff().fillna(0).round(2)
    weather_df["Extreme_Weather_Flag"] = (
        (weather_df["Temperature_F"] >= 95)
        | (weather_df["Rainfall_In"] >= 0.25)
        | (weather_df["Temperature_Change_1h_F"].abs() >= 10)
    ).astype(int)
    weather_df = weather_df[
        ["DateTime_Hour"] + WEATHER_COLUMNS + ["Weather_Code"]
    ]

    ensure_directory(cache_path.parent)
    weather_df.to_csv(cache_path, index=False)
    return weather_df


def add_calendar_features(feature_df: pd.DataFrame) -> pd.DataFrame:
    df = feature_df.copy()
    df["DateTime_Hour"] = pd.to_datetime(df["DateTime_Hour"])
    df = df.sort_values("DateTime_Hour").reset_index(drop=True)
    calendar = USFederalHolidayCalendar()

    holiday_dates = pd.DatetimeIndex(
        calendar.holidays(
            start=df["DateTime_Hour"].min().normalize(),
            end=df["DateTime_Hour"].max().normalize(),
        )
    ).normalize()
    day_after_holiday = holiday_dates + pd.Timedelta(days=1)
    dates = df["DateTime_Hour"].dt.normalize()

    df["Hour"] = df["DateTime_Hour"].dt.hour
    df["Day_Of_Week"] = df["DateTime_Hour"].dt.day_name()
    df["Is_Weekend"] = (df["DateTime_Hour"].dt.dayofweek >= 5).astype(int)
    df["Month"] = df["DateTime_Hour"].dt.month_name()
    df["Holiday_Flag"] = dates.isin(holiday_dates).astype(int)
    df["Day_After_Holiday"] = dates.isin(day_after_holiday).astype(int)

    return df


def add_external_features(feature_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    df = feature_df.copy()
    df["Is_Flu_Season"] = df["DateTime_Hour"].dt.month.isin([10, 11, 12, 1, 2, 3]).astype(int)
    weather_df = weather_df.copy()
    weather_df["DateTime_Hour"] = pd.to_datetime(weather_df["DateTime_Hour"])
    return df.merge(
        weather_df[["DateTime_Hour"] + WEATHER_COLUMNS], on="DateTime_Hour", how="left"
    )


def add_lag_features(feature_df: pd.DataFrame, lag_hours: Iterable[int] = LAG_HOURS) -> pd.DataFrame:
    df = feature_df.copy()
    for lag in lag_hours:
        df[f"Lag_{lag}h"] = df["Patient_Count"].shift(lag)
    return df


def add_rolling_features(feature_df: pd.DataFrame) -> pd.DataFrame:
    df = feature_df.copy()
    shifted = df["Patient_Count"].shift(1)
    for window in ROLLING_MEAN_WINDOWS:
        df[f"Rolling_Mean_{window}h"] = shifted.rolling(window).mean()
    for window in ROLLING_STD_WINDOWS:
        df[f"Rolling_Std_{window}h"] = shifted.rolling(window).std()
    return df


def add_cyclical_features(feature_df: pd.DataFrame) -> pd.DataFrame:
    df = feature_df.copy()
    df["Day_Of_Week_Num"] = df["DateTime_Hour"].dt.dayofweek
    df["Month_Num"] = df["DateTime_Hour"].dt.month
    df["Hour_Sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["Hour_Cos"] = np.cos(2 * np.pi * df["Hour"] / 24)
    df["Day_Of_Week_Sin"] = np.sin(2 * np.pi * df["Day_Of_Week_Num"] / 7)
    df["Day_Of_Week_Cos"] = np.cos(2 * np.pi * df["Day_Of_Week_Num"] / 7)
    df["Month_Sin"] = np.sin(2 * np.pi * df["Month_Num"] / 12)
    df["Month_Cos"] = np.cos(2 * np.pi * df["Month_Num"] / 12)
    return df


def build_feature_table(
    regularized_df: pd.DataFrame,
    weather_df: pd.DataFrame | None = None,
    weather_cache_path: Path | str | None = None,
) -> pd.DataFrame:
    df = regularized_df.copy()
    df["DateTime_Hour"] = pd.to_datetime(df["DateTime_Hour"])
    df = df.sort_values("DateTime_Hour").reset_index(drop=True)

    df = add_calendar_features(df)
    if weather_df is None:
        weather_df = load_or_fetch_los_angeles_weather(
            start_timestamp=df["DateTime_Hour"].min(),
            end_timestamp=df["DateTime_Hour"].max(),
            cache_path=weather_cache_path,
        )
    df = add_external_features(df, weather_df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_cyclical_features(df)

    modeling_df = df.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    return modeling_df
