import yfinance as yf
import pandas as pd

print("--- Fetching Sector ETF Historical Data via yfinance ---")

sectors = ['XLF', 'XLK', 'XLE', 'XLV', 'XLY', 'XLP', 'XLI', 'XLU', 'XLB']
fetch_start_date = "2004-12-01"

raw_data = yf.download(sectors, start=fetch_start_date, auto_adjust=True)['Close']

monthly_prices = raw_data.resample('MS').last()
monthly_returns = monthly_prices.pct_change() * 100
monthly_returns = monthly_returns.round(2).reset_index().rename(columns={'Date': 'date'})
monthly_returns['date'] = monthly_returns['date'].dt.strftime('%Y-%m-%d')
monthly_returns = monthly_returns[monthly_returns['date'] >= '2005-01-01']

desired_columns = ['date', 'XLF', 'XLK', 'XLE', 'XLV', 'XLY', 'XLP', 'XLI', 'XLU', 'XLB']
monthly_returns = monthly_returns[desired_columns]

monthly_returns.to_csv('data/sector_returns.csv', index=False)
print("\nSuccessfully fetched, calculated, and saved sector returns to 'data/sector_returns.csv'!\n")

print("--- FIRST 5 ROWS ---")
print(monthly_returns.head(5).to_string(index=False))
print("\n--- LAST 5 ROWS ---")
print(monthly_returns.tail(5).to_string(index=False))
