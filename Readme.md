# 🚑 Emergency Room Demand Forecasting

## 🎯 Goal
Build a robust multi-horizon forecasting system for Emergency Room (ER) volumes and triage acuity distributions. The project models volatility across:
- **Short-term**: hourly / daily dispatch planning
- **Medium-term**: monthly seasonal surges
- **Long-term**: annual demographic and trend changes

The aim is to reduce bottlenecks, optimize staffing, and align critical care resources with demand before patient surges occur.

## 🧠 Approach
Forecasting is treated as a multi-stage time-series problem, with modeling tailored to each planning horizon.

### Short-term forecasting (1–72 hours)
- Capture high-frequency volatility and irregular arrival spikes
- Use models such as **LightGBM** or recurrent architectures like **LSTM**
- Model intra-day patterns, weekday/weekend seasonality, and acuity shifts

### Medium- and long-term forecasting (1–12+ months)
- Use structural time-series and advanced statistical models
- Explore **SARIMAX**, **Prophet**, or **state-space models**
- Focus on broader signals such as influenza seasonality and demographic trends

## 📦 Data
The project uses operational emergency department records, including:
- Timestamped patient arrivals
- Treatment outcomes and lengths of stay
- Patient demographics and triage categories

This information is structured in the dataset file and is critical for forecasting both volume and severity of incoming demand.

## ✅ Success Metrics
Model performance is evaluated using domain-appropriate error measures:
- **RMSE**: penalize large deviations in hourly/daily forecasts
- **MAE**: ensure stable short-term prediction accuracy
- **MAPE**: evaluate long-term trend accuracy

### Business impact
The forecast must also demonstrate real-world value by supporting:
- Reduced average patient wait times during peak demand
- Lower reliance on emergency overflow staffing contracts
- More stable patient-to-provider ratios across planning horizons

## 📁 Project structure
- `data/raw/` — raw dataset files downloaded for modeling
- `Notebook/` — analysis and modeling notebook(s)
- `Readme.md` — project overview and methodology

## 🚀 Notes
Make sure the Kaggle CLI is configured and the dataset is available in `data/raw/` before running the modeling workflow.

