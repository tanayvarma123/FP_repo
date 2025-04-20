# fetch_data.py

import pandas as pd
from datetime import datetime

tickers = {
    'oil': 'OILPRODUS',
    'gdp': 'GDPUS',
    'indprod': 'IPUS',
    'private_cons': 'RPRCUS',
    'cpi': 'CPIUS',
    'ppi': 'PPIUS',
    'capform': 'RGFCFUS',
    'cab': 'CAUS',
    'yield10y': 'Y10YDUS'
}

def fetch_data(ticker):
    url = f"https://www.econdb.com/api/series/{ticker}/?format=csv&frequency=M&token=92a25975a3c9c874cc143b70c6066ff4a95378c9"
    try:
        df = pd.read_csv(url, index_col='Date', parse_dates=['Date'])
        df.columns = [ticker]
        return df
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return pd.DataFrame()

datasets = [fetch_data(tkr) for tkr in tickers.values()]
df = pd.concat(datasets, axis=1)

# Convert to dd-mm-yyyy format
df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%d-%m-%Y')
# Filter rows to keep dates after 2002
df = df[pd.to_datetime(df['Date'], format='%d-%m-%Y') >= '01-01-2002']
# Oil Production data is not available for the last 2 months, so dropping them as na
df = df.dropna(subset=['OILPRODUS'])
# Producer price index data is not available before 2014, so we are dropping that column
df.drop(columns=['PPIUS'], inplace=True)
df = df.set_index('Date', drop=True)
df = df.interpolate(method='linear', inplace=False)

# Save as CSV
df.to_csv("economic_data.csv")

# Append timestamp to ensure change is detected
with open("economic_data.csv", "a") as f:
    f.write(f"# Updated on {datetime.utcnow().isoformat()} UTC\n")

print("economic_data.csv saved.")
