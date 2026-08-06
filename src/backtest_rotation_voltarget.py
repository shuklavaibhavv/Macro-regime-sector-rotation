import pandas as pd
import numpy as np

print("--- Volatility-Targeted Out-of-Sample Sector Rotation Backtest ---")

df = pd.read_csv("data/regime_sector_merged.csv")
df['regime_lagged'] = df['regime'].shift(1)
df = df.dropna(subset=['regime_lagged']).reset_index(drop=True)

split_date = "2016-01-01"

train_df = df[df['date'] < split_date].copy()
sectors = ['XLF', 'XLK', 'XLE', 'XLV', 'XLY', 'XLP', 'XLI', 'XLU', 'XLB']

train_avg_returns = train_df.groupby('regime_lagged')[sectors].mean()
train_allocation_map = {regime: train_avg_returns.loc[regime].idxmax() for regime in train_avg_returns.index}

df['chosen_sector'] = df['regime_lagged'].map(train_allocation_map)
df['unscaled_return'] = df.apply(lambda row: row[row['chosen_sector']], axis=1)

target_volatility = 15.0

test_mask = df['date'] >= split_date
test_indices = df[test_mask].index

scaled_returns = []

for idx in test_indices:
    sector = df.loc[idx, 'chosen_sector']
    trailing_12m_returns = df.loc[idx-12 : idx-1, sector]
    trailing_vol = trailing_12m_returns.std() * np.sqrt(12)
    
    if trailing_vol == 0 or np.isnan(trailing_vol):
        position_scalar = 1.0
    else:
        position_scalar = target_volatility / trailing_vol
    
    position_scalar = np.clip(position_scalar, 0.3, 1.5)
    month_scaled_return = df.loc[idx, sector] * position_scalar
    scaled_returns.append(month_scaled_return)

test_df = df[test_mask].copy()
test_df['scaled_return'] = scaled_returns
test_df['benchmark_return'] = test_df[sectors].mean(axis=1)

test_df['cum_unscaled'] = (1 + test_df['unscaled_return'] / 100).cumprod()
test_df['cum_scaled'] = (1 + test_df['scaled_return'] / 100).cumprod()
test_df['cum_benchmark'] = (1 + test_df['benchmark_return'] / 100).cumprod()

n_test_months = len(test_df)
test_years = n_test_months / 12.0

def calculate_performance_metrics(monthly_returns, cum_series):
    total_return = (cum_series.iloc[-1] - 1) * 100
    cagr = ((cum_series.iloc[-1]) ** (1 / test_years) - 1) * 100
    ann_volatility = monthly_returns.std() * np.sqrt(12)
    sharpe_ratio = cagr / ann_volatility if ann_volatility != 0 else np.nan
    
    running_peak = cum_series.cummax()
    drawdown_series = (cum_series - running_peak) / running_peak
    max_drawdown = drawdown_series.min() * 100
    
    return {
        'Total Return (%)': round(total_return, 2),
        'CAGR (%)': round(cagr, 2),
        'Ann. Volatility (%)': round(ann_volatility, 2),
        'Sharpe Ratio': round(sharpe_ratio, 2),
        'Max Drawdown (%)': round(max_drawdown, 2)
    }

unscaled_metrics = calculate_performance_metrics(test_df['unscaled_return'], test_df['cum_unscaled'])
scaled_metrics = calculate_performance_metrics(test_df['scaled_return'], test_df['cum_scaled'])
benchmark_metrics = calculate_performance_metrics(test_df['benchmark_return'], test_df['cum_benchmark'])

comparison_table = pd.DataFrame([unscaled_metrics, scaled_metrics, benchmark_metrics], index=[
    'OOS Unscaled Sector Rotation',
    'OOS Volatility-Scaled Rotation (15% Target)',
    'OOS Equal-Weight Benchmark'
])

print("==========================================================================================")
print(f"        THREE-WAY OUT-OF-SAMPLE PERFORMANCE COMPARISON ({test_years:.1f} YEARS: 2016-PRESENT)     ")
print("==========================================================================================")
print(comparison_table.to_string())
