# Execution

## Setup

1. Create a Python environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Run the pipeline

```bash
python -m src.train
```

## Data expectations

- Raw data should be stored in `data/raw/Hospital ER_Data.csv`.
- The pipeline writes `data/hourly_patient_volume.csv` and `data/hourly_patient_volume_regular.csv`.
- Cached weather data is stored in `data/weather_hourly_los_angeles.csv`.

## Testing

Run tests with:

```bash
pytest
```
