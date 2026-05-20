from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .config import load_config
from .data_loader import (
    DEFAULT_HOURLY_PATH,
    DEFAULT_REGULAR_PATH,
    build_hourly_dataset,
    load_hourly_data,
    load_raw_data,
    regularize_hourly_data,
)
from .evaluate import build_metrics_frame
from .preprocessing import (
    FEATURE_COLUMNS,
    FORECAST_HORIZONS,
    build_feature_table,
    load_or_fetch_los_angeles_weather,
)
from .utils import clip_nonnegative, set_random_seed


def create_supervised_targets(
    feature_df: pd.DataFrame,
    horizons: dict[int, str] = FORECAST_HORIZONS,
) -> tuple[pd.DataFrame, list[str]]:
    supervised_df = feature_df.copy()
    for horizon, target_column in horizons.items():
        supervised_df[target_column] = supervised_df["Patient_Count"].shift(-horizon)
    target_columns = list(horizons.values())
    supervised_df = supervised_df.dropna(subset=target_columns).reset_index(drop=True)
    return supervised_df, target_columns


def encode_features(
    modeling_df: pd.DataFrame,
    feature_columns: list[str] = FEATURE_COLUMNS,
    target_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    feature_columns = list(feature_columns)
    target_columns = target_columns or [col for col in modeling_df.columns if col.startswith("Target_")]
    X_encoded = pd.get_dummies(modeling_df[feature_columns], drop_first=False, dtype=float)
    y_targets = modeling_df[target_columns].copy()
    forecast_origins = modeling_df["DateTime_Hour"].copy()
    return X_encoded, y_targets, forecast_origins


def split_chronological(
    X: pd.DataFrame,
    y: pd.DataFrame,
    forecast_origins: pd.Series,
    train_ratio: float,
    validation_ratio: float,
) -> dict[str, Any]:
    n = len(X)
    train_end = int(n * train_ratio)
    validation_end = int(n * (train_ratio + validation_ratio))

    return {
        "X_train": X.iloc[:train_end],
        "X_val": X.iloc[train_end:validation_end],
        "X_train_val": X.iloc[:validation_end],
        "X_test": X.iloc[validation_end:],
        "y_train": y.iloc[:train_end],
        "y_val": y.iloc[train_end:validation_end],
        "y_train_val": y.iloc[:validation_end],
        "y_test": y.iloc[validation_end:],
        "date_train": forecast_origins.iloc[:train_end],
        "date_val": forecast_origins.iloc[train_end:validation_end],
        "date_test": forecast_origins.iloc[validation_end:],
        "train_end": train_end,
        "validation_end": validation_end,
    }


def build_baseline_predictions(
    modeling_supervised_df: pd.DataFrame,
    y_test: pd.DataFrame,
    target_columns: list[str],
) -> dict[str, pd.DataFrame]:
    baseline_predictions: dict[str, pd.DataFrame] = {
        "Persistence": pd.DataFrame(index=y_test.index),
        "Seasonal Naive 24h": pd.DataFrame(index=y_test.index),
        "Seasonal Naive 168h": pd.DataFrame(index=y_test.index),
        "Moving Average 24h": pd.DataFrame(index=y_test.index),
    }

    for target_column in target_columns:
        baseline_predictions["Persistence"][target_column] = (
            modeling_supervised_df.loc[y_test.index, "Patient_Count"].values
        )
        baseline_predictions["Seasonal Naive 24h"][target_column] = (
            modeling_supervised_df.loc[y_test.index, "Lag_24h"].values
        )
        baseline_predictions["Seasonal Naive 168h"][target_column] = (
            modeling_supervised_df.loc[y_test.index, "Lag_168h"].values
        )
        baseline_predictions["Moving Average 24h"][target_column] = (
            modeling_supervised_df.loc[y_test.index, "Rolling_Mean_24h"].values
        )

    for predictions in baseline_predictions.values():
        predictions[:] = clip_nonnegative(predictions)

    return baseline_predictions


def train_regression_models(
    X_train_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train_val: pd.DataFrame,
    y_test: pd.DataFrame,
    target_columns: list[str],
    config: dict[str, Any],
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    model_specs = {
        "Poisson Regression": lambda: make_pipeline(
            StandardScaler(),
            PoissonRegressor(
                alpha=float(config["regression"]["poisson"]["alpha"]),
                max_iter=int(config["regression"]["poisson"]["max_iter"]),
            ),
        ),
        "Ridge Regression": lambda: make_pipeline(
            StandardScaler(),
            Ridge(alpha=float(config["regression"]["ridge"]["alpha"])),
        ),
    }

    prediction_store: dict[str, pd.DataFrame] = {}
    rows = []

    for model_name, construct in model_specs.items():
        model_predictions = pd.DataFrame(index=y_test.index)
        for horizon, target_column in FORECAST_HORIZONS.items():
            model = construct()
            model.fit(X_train_val, y_train_val[target_column])
            predictions = clip_nonnegative(model.predict(X_test))
            model_predictions[target_column] = predictions
            rows.append(
                {
                    "Model": model_name,
                    "Model_Group": "Interpretable Regression",
                    "Horizon": f"{horizon}h",
                    **build_metrics_frame({model_name: model_predictions}, y_test, {model_name: f"Interpretable Regression"}, FORECAST_HORIZONS).iloc[-1].to_dict(),
                }
            )

        prediction_store[model_name] = model_predictions

    metrics_df = pd.DataFrame(rows)
    return prediction_store, metrics_df


def train_sarimax_model(
    modeling_supervised_df: pd.DataFrame,
    y_test: pd.DataFrame,
    date_test: pd.Series,
    validation_end: int,
    config: dict[str, Any],
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, list[str]]:
    sarimax_exog_columns = [
        "Hour_Sin",
        "Hour_Cos",
        "Day_Of_Week_Sin",
        "Day_Of_Week_Cos",
        "Is_Weekend",
        "Is_Flu_Season",
        "Temperature_F",
        "Rainfall_In",
        "Extreme_Weather_Flag",
        "Rolling_Mean_24h",
    ]
    sarimax_exog_columns = [col for col in sarimax_exog_columns if col in modeling_supervised_df.columns]
    sarimax_train_val = modeling_supervised_df.iloc[:validation_end].copy()
    sarimax_test = modeling_supervised_df.iloc[validation_end:].copy()
    sarimax_training_tail = min(3000, len(sarimax_train_val))
    sarimax_train_tail = sarimax_train_val.tail(sarimax_training_tail)

    try:
        sarimax_model = SARIMAX(
            endog=sarimax_train_tail["Target_1h"].astype(float),
            exog=sarimax_train_tail[sarimax_exog_columns].astype(float),
            order=tuple(config["sarimax"]["order"]),
            seasonal_order=tuple(config["sarimax"]["seasonal_order"]),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        sarimax_result = sarimax_model.fit(disp=False, maxiter=int(config["sarimax"]["maxiter"]))
        sarimax_forecast = sarimax_result.forecast(
            steps=len(sarimax_test),
            exog=sarimax_test[sarimax_exog_columns].astype(float),
        )
        predictions = clip_nonnegative(sarimax_forecast)
        sarimax_df = pd.DataFrame(index=y_test.index)
        sarimax_df["Target_1h"] = predictions

        metrics_df = pd.DataFrame(
            [
                {
                    "Model": "SARIMAX",
                    "Model_Group": "Time-Series Regression",
                    "Horizon": "1h",
                    **build_metrics_frame({"SARIMAX": sarimax_df}, y_test, {"SARIMAX": "Time-Series Regression"}, FORECAST_HORIZONS).iloc[0].to_dict(),
                }
            ]
        )
        return {"SARIMAX": sarimax_df}, metrics_df, []
    except Exception as error:
        return {}, pd.DataFrame(), [f"SARIMAX skipped: {error}"]


def train_ml_models(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_train_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_val: pd.DataFrame,
    y_train_val: pd.DataFrame,
    y_test: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    try:
        import lightgbm as lgb
    except ImportError:
        lgb = None

    model_name = "LightGBM" if lgb is not None else "HistGradientBoosting"
    model_predictions = pd.DataFrame(index=y_test.index)
    rows = []

    for horizon, target_column in FORECAST_HORIZONS.items():
        if lgb is not None:
            model = lgb.LGBMRegressor(
                objective="regression",
                n_estimators=int(config["lightgbm"]["n_estimators"]),
                learning_rate=float(config["lightgbm"]["learning_rate"]),
                num_leaves=int(config["lightgbm"]["num_leaves"]),
                subsample=float(config["lightgbm"]["subsample"]),
                colsample_bytree=float(config["lightgbm"]["colsample_bytree"]),
                min_child_samples=int(config["lightgbm"]["min_child_samples"]),
                random_state=int(config["seed"]),
                verbose=-1,
            )
            model.fit(
                X_train,
                y_train[target_column],
                eval_set=[(X_val, y_val[target_column])],
                eval_metric="l2",
                callbacks=[
                    lgb.early_stopping(int(config["lightgbm"]["early_stopping_rounds"]), verbose=False),
                    lgb.log_evaluation(0),
                ],
            )
        else:
            model = HistGradientBoostingRegressor(
                learning_rate=float(config["hist_gradient_boosting"]["learning_rate"]),
                max_iter=int(config["hist_gradient_boosting"]["max_iter"]),
                l2_regularization=float(config["hist_gradient_boosting"]["l2_regularization"]),
                random_state=int(config["seed"]),
            )
            model.fit(X_train_val, y_train_val[target_column])

        model_predictions[target_column] = clip_nonnegative(model.predict(X_test))
        rows.append(
            {
                "Model": model_name,
                "Model_Group": "Nonlinear ML",
                "Horizon": f"{horizon}h",
                **build_metrics_frame({model_name: model_predictions}, y_test, {model_name: "Nonlinear ML"}, FORECAST_HORIZONS).iloc[-1].to_dict(),
            }
        )

    return {model_name: model_predictions}, pd.DataFrame(rows)


def main(config_path: Path | str | None = None) -> None:
    config = load_config(config_path)
    training_config = load_config(config.get("training_config", "configs/training.yaml"))
    set_random_seed(int(config.get("seed", 42)))

    hourly_csv = Path(config["data"]["hourly_csv"])
    regular_csv = Path(config["data"]["regular_csv"])

    if hourly_csv.exists():
        hourly_df = load_hourly_data(hourly_csv)
    else:
        raw_df = load_raw_data()
        hourly_df = regularize_hourly_data(build_hourly_dataset(raw_df, output_path=hourly_csv), output_path=regular_csv)

    if regular_csv.exists():
        regular_df = pd.read_csv(regular_csv)
        regular_df["DateTime_Hour"] = pd.to_datetime(regular_df["DateTime_Hour"])
    else:
        regular_df = regularize_hourly_data(hourly_df, output_path=regular_csv)

    feature_df = build_feature_table(
        regular_df,
        weather_cache_path=Path(config["data"]["weather_csv"]),
    )
    modeling_supervised_df, target_columns = create_supervised_targets(feature_df)
    X_encoded, y_targets, forecast_origins = encode_features(modeling_supervised_df, FEATURE_COLUMNS, target_columns)

    splits = split_chronological(
        X_encoded,
        y_targets,
        forecast_origins,
        train_ratio=float(training_config["train_ratio"]),
        validation_ratio=float(training_config["validation_ratio"]),
    )

    baseline_predictions = build_baseline_predictions(modeling_supervised_df, splits["y_test"], target_columns)
    baseline_metrics = build_metrics_frame(
        baseline_predictions,
        splits["y_test"],
        {model_name: "Simple Benchmark" for model_name in baseline_predictions},
        FORECAST_HORIZONS,
    )

    regression_predictions, regression_metrics = train_regression_models(
        splits["X_train_val"],
        splits["X_test"],
        splits["y_train_val"],
        splits["y_test"],
        target_columns,
        training_config,
    )

    sarimax_predictions, sarimax_metrics, sarimax_notes = train_sarimax_model(
        modeling_supervised_df,
        splits["y_test"],
        splits["date_test"],
        splits["validation_end"],
        training_config["sarimax"],
    )

    ml_predictions, ml_metrics = train_ml_models(
        splits["X_train"],
        splits["X_val"],
        splits["X_train_val"],
        splits["X_test"],
        splits["y_train"],
        splits["y_val"],
        splits["y_train_val"],
        splits["y_test"],
        training_config,
    )

    all_metrics = pd.concat([baseline_metrics, regression_metrics, sarimax_metrics, ml_metrics], ignore_index=True)
    best_by_horizon = all_metrics.loc[all_metrics.groupby("Horizon")["RMSE"].idxmin()].reset_index(drop=True)
    print("Best model by horizon:")
    print(best_by_horizon.to_string(index=False))

    if sarimax_notes:
        print("Model notes:")
        for note in sarimax_notes:
            print(f"- {note}")


if __name__ == "__main__":
    main()
