import pandas as pd
import joblib
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("economic_data.csv", header=True)


## DATA CLEANING ##

# Convert to dd-mm-yyyy format
df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%d-%m-%Y')

# Filter rows to keep dates after 2002
df = df[pd.to_datetime(df['Date'], format='%d-%m-%Y') >= '01-01-2002']

# Oil Production data is not available for the last 2 months, so dropping them as na
df = df.dropna(subset=['OILPRODUS'])

# Producer price index data is not available before 2014, so we are dropping that column
df.drop(columns=['PPIUS'], inplace=True)
df = df.set_index('Date', drop=True)

# Fit SARIMA model with best order
arima_model = SARIMAX(df['OILPRODUS'], order=(9,1,10))
arima_result = arima_model.fit(disp=False)

# Save ARIMA model
joblib.dump(arima_result, 'arima_model.pkl')
print("Model saved as 'arima_model.pkl'")