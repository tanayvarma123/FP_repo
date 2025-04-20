import pandas as pd
import io
import joblib
from statsmodels.tsa.statespace.sarimax import SARIMAX
from azure.storage.blob import BlobServiceClient
import warnings
warnings.filterwarnings("ignore")
from pandas.tseries.offsets import DateOffset

# Azure Blob Storage setup
connection_string = "DefaultEndpointsProtocol=https;AccountName=foundationprojectstorage;AccountKey=GNr/K5ligRjMCu+G+XaZDrFxw1axPdd9zlHxAkbNvvgdWhYTfU3pK2XjLkKD0w07jnfOuiTndZOz+AStFjXgTQ==;EndpointSuffix=core.windows.net"
blob_service_client = BlobServiceClient.from_connection_string(connection_string)
raw_data_container = "raw-data"

def download_from_blob(blob_name):
    # Create a BlobClient
    blob_client = blob_service_client.get_blob_client(container=raw_data_container, blob=blob_name)
    
    # Download the blob content as a string
    blob_data = blob_client.download_blob().content_as_text()
    
    # Convert the string data to a DataFrame
    df = pd.read_csv(io.StringIO(blob_data))
    return df

df = download_from_blob("economic_data.csv")
df.head()

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

# Ensure datetime index
df.index = pd.to_datetime(df.index, format="%d-%m-%Y")  # convert from string to datetime

# Forecast
forecast_6 = arima_result.forecast(steps=6)
forecast_48 = arima_result.forecast(steps=48)

# ----------------------------
# Create datetime indices for forecasts
last_date = df.index[-1]
index_6 = [last_date + DateOffset(months=i) for i in range(1, 7)]
index_48 = [last_date + DateOffset(months=i) for i in range(1, 49)]

forecast_6.index = index_6
forecast_48.index = index_48

# Save ARIMA model
joblib.dump(arima_result, 'arima_model.pkl')
print("Model saved as 'arima_model.pkl'")