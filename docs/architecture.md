# Architecture

This repository separates the forecasting workflow into discrete modules:

- `src/data_loader.py`: Loads raw patient encounter data and builds the reproducible hourly forecasting dataset.
- `src/preprocessing.py`: Adds calendar, weather, lag, rolling, and cyclical features to create a modeling-ready dataset.
- `src/train.py`: Defines chronological splits, baseline and model training, and best-model selection.
- `src/evaluate.py`: Computes model performance metrics, operational validation summaries, and comparison tables.
- `src/predict.py`: Provides inference helpers for applying trained models to new feature data.
- `src/config.py`: Loads YAML configuration so experiments are reproducible and pipeline settings are version controlled.
