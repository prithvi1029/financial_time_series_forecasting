# 📊 Financial Product Forecasting Framework with Explainable AI

This end-to-end project showcases a robust AI-driven pipeline for forecasting financial product demand by integrating macroeconomic factors and using both deep learning (LSTM) and machine learning (XGBoost). The pipeline includes synthetic data generation, time series forecasting, feature attribution with SHAP, and 6-month demand prediction.

---

## 🧠 Modules

### 1. `Macro_economic_factors_data_generation.ipynb`

Generates a realistic synthetic dataset of 29 macroeconomic indicators from Jan 2019 to Feb 2025.

- Trend + Seasonality + Noise generation for:
  - GDP Growth, CPI, Interest Rates, Oil Price, S&P500, Debt-to-GDP, Fintech, Trade Balance, etc.
- Saves dataset to: `macro_economic_factors.csv`

---

### 2. `Financial_time_series_forecasting_LSTM_base_model.ipynb`

Builds and trains an LSTM model for multivariate time series forecasting.

- Uses past 9 months to predict the next month's values.
- LSTM with 2 layers + Dropout, trained with `MAPE` loss.
- Validation and test set evaluation.
- Final 6-month forecast (Feb 2025 – Jul 2025) saved to `6_month_forecast.csv`
- Model saved as: `lstm_macro_forecast_final.h5`

---

### 3. `xgboost_sales_forecast.ipynb`

Forecasts future sales for a selected product (e.g., PQ001) using macroeconomic forecasts and interpretable ML.

- Loads macro + product sales data.
- Merges and filters relevant features.
- Performs train/validation/test splits.
- Uses XGBoost with `Hyperopt` for hyperparameter tuning.
- Evaluates model on test set using RMSE.
- Saves model: `best_model.pkl`

**Explainability & Insights:**
- Feature importance using XGBoost built-in scores.
- SHAP value plots (bar plots and per-instance SHAP bars).
- Saves SHAP results to `shap_df`.

---

## 🔁 Pipeline Flow

1. **Generate Macro Data** → 2. **Train LSTM Forecasting Model** → 3. **Forecast 6 Months of Macro Trends**  
4. **Merge with Sales Data** → 5. **Train XGBoost on Product Sales** → 6. **Explain using SHAP**

---

## 📁 Key Files

| File                           | Description                                      |
|--------------------------------|--------------------------------------------------|
| `macro_economic_factors.csv`  | Synthetic macroeconomic indicators (monthly)     |
| `6_month_forecast.csv`        | LSTM-based macro forecasts for 6 months          |
| `lstm_macro_forecast_final.h5`| Trained LSTM forecasting model                   |
| `best_model.pkl`              | Final XGBoost sales forecasting model            |
| `shap_df`                     | SHAP value DataFrame                             |

---

## 📦 Requirements

```bash
pip install pandas numpy tensorflow scikit-learn matplotlib shap xgboost hyperopt
