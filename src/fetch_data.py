# Import pandas library for manipulating data tables (DataFrames)
import pandas as pd

# Import numpy library for handling numerical missing values (NaN)
import numpy as np

# Import ssl module to handle secure HTTPS downloads on macOS without SSL certificate errors
import ssl

# Fix SSL context issues on macOS when downloading data from web URLs
ssl._create_default_https_context = ssl._create_unverified_context

# Print header to signal script start
print("--- Fetching Macroeconomic Data from FRED ---")

# Define the start year for our historical dataset
start_date = "2005-01-01"

# -------------------------------------------------------------------
# 1. PULL YIELD CURVE SPREAD DATA (T10Y2Y) FROM FRED
# -------------------------------------------------------------------
url_yield = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=T10Y2Y"
df_yield = pd.read_csv(url_yield)
df_yield['DATE'] = pd.to_datetime(df_yield['observation_date'])
df_yield = df_yield.rename(columns={'T10Y2Y': 'yield_curve_spread'})
df_yield['yield_curve_spread'] = pd.to_numeric(df_yield['yield_curve_spread'], errors='coerce')
df_yield = df_yield.set_index('DATE')
df_yield_monthly = df_yield['yield_curve_spread'].resample('MS').mean()

# -------------------------------------------------------------------
# 2. PULL ISM PMI PROXY DATA (IPMAN) FROM FRED
# -------------------------------------------------------------------
url_ipman = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=IPMAN"
df_ipman = pd.read_csv(url_ipman)
df_ipman['DATE'] = pd.to_datetime(df_ipman['observation_date'])
df_ipman['IPMAN'] = pd.to_numeric(df_ipman['IPMAN'], errors='coerce')
df_ipman = df_ipman.set_index('DATE')
df_ipman_monthly = df_ipman['IPMAN'].resample('MS').mean()

ipman_yoy = df_ipman_monthly.pct_change(periods=12) * 100
df_pmi_proxy = 50 + ipman_yoy

# -------------------------------------------------------------------
# 3. IDENTIFY GENUINE END DATES BEFORE FORWARD-FILLING
# -------------------------------------------------------------------
last_pmi_date = df_pmi_proxy.dropna().index.max()
last_yield_date = df_yield_monthly.dropna().index.max()
cutoff_date = min(last_pmi_date, last_yield_date)

# -------------------------------------------------------------------
# 4. MERGE, FILTER, AND TRIM THE DATAFRAME
# -------------------------------------------------------------------
macro_df = pd.DataFrame({
    'pmi': df_pmi_proxy,
    'yield_curve_spread': df_yield_monthly
})

macro_df = macro_df[macro_df.index >= start_date]
macro_df = macro_df[macro_df.index <= cutoff_date]
macro_df = macro_df.ffill().dropna().round(2).reset_index().rename(columns={'DATE': 'date'})

# -------------------------------------------------------------------
# 5. SAVE TO DATA FOLDER & DISPLAY PREVIEW
# -------------------------------------------------------------------
macro_df.to_csv('data/macro_data.csv', index=False)

print("Trimming Details:")
print(f"  - Last genuine PMI data date:   {last_pmi_date.strftime('%Y-%m-%d')}")
print(f"  - Last genuine Yield data date: {last_yield_date.strftime('%Y-%m-%d')}")
print(f"  - Dataset trimmed to end at:   {cutoff_date.strftime('%Y-%m-%d')} (to avoid trailing ffill flat spots)")

print("\nSuccessfully fetched, processed, trimmed, and saved macro data to 'data/macro_data.csv'!\n")
print("--- FIRST 10 ROWS ---")
print(macro_df.head(10))
print("\n--- LAST 5 ROWS ---")
print(macro_df.tail(5))
