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

# Reset index so 'Date' is a column
df.reset_index(inplace=True)

# Ensure 'Date' is in datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Filter to keep dates from 2002 onwards
df = df[df['Date'] >= pd.to_datetime('2002-01-01')]

# Drop rows with missing oil data
df = df.dropna(subset=['OILPRODUS'])

# Drop PPIUS column if it exists
if 'PPIUS' in df.columns:
    df.drop(columns=['PPIUS'], inplace=True)

# Set Date as index and interpolate
df = df.set_index('Date')
df = df.interpolate(method='linear', inplace=False)

# Save as CSV
df.to_csv("economic_data.csv")

# Append update timestamp
with open("economic_data.csv", "a") as f:
    f.write(f"# Updated on {datetime.utcnow().isoformat()} UTC\n")

print("economic_data.csv saved.")
