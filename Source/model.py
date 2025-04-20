# score.py

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
import joblib

# ----------------------------
# Load data

df = pd.read_csv("Data/economic_data.csv", index_col='Date', parse_dates=['Date'])
series = df['OILPRODUS'].dropna()

# ----------------------------
# Fit ARIMA model
model = ARIMA(series, order=(9, 1, 10))
model_fit = model.fit()

# ----------------------------
# Calculate model performance
fitted_values = model_fit.fittedvalues
rmse = mean_squared_error(series[1:], fitted_values[1:], squared=False)
aic = model_fit.aic

print(f"✅ Model Performance: AIC = {aic:.2f}, RMSE = {rmse:.2f}")

# ----------------------------
# Save model
joblib.dump(model_fit, "Data/arima_model.pkl")
print("✅ ARIMA model saved as Data/arima_model.pkl")
