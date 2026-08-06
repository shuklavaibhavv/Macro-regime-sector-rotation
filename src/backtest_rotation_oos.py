import pandas as pd
import numpy as np

print("--- Out-of-Sample (OOS) Sector Rotation Backtest ---")

df = pd.read_csv("data/regime_sector_merged.csv")
df['regime_lagged'] = df['regime'].shift(1)
df = df.dropna(subset=['regime_lagged']).copy()

split_date = "2016-01-01"

train_df = df[df['date'] < split_date].copy()
test_df = df[df['date'] >= split_date].copy()

print(f"Dataset Split Summary:")
print(f"  • Training Period (In-Sample):     {len(train_df)} months (2005-01-01 to 2015-12-01)")
print(f"  • Test Period     (Out-of-Sample): {len(test_df)} months (2016-01-01 to Present)\n")

sectors = ['XLF', 'XLK', 'XLE', 'XLV', 'XLY', 'XLP', 'XLI', 'XLU', 'XLB']
train_avg_returns = train_df.groupby('regime_lagged')[sectors].mean()

train_allocation_map = {}
for regime in train_avg_returns.index:
    best_sector = train_avg_returns.loc[regime].idxmax()
    train_allocation_map[regime] = best_sector

print("=== PROGRAMMATICALLY DERIVED TRAINING ALLOCATION MAP (2005 - 2015) ===")
for regime, sector in train_allocation_map.items():
    avg_ret = train_avg_returns.loc[regime, sector]
    print(f"  • {regime:12s} -> {sector} ({avg_ret:.2f}% avg monthly return in training)")

test_df['chosen_sector'] = test_df['regime_lagged'].map(train_allocation_map)
test_df['strategy_return'] = test_df.apply(lambda row: row[row['chosen_sector']], axis=1)
test_df['benchmark_return'] = test_df[sectors].mean(axis=1)

test_df['cumulative_strategy'] = (1 + test_df['strategy_return'] / 100).cumprod()
test_df['cumulative_benchmark'] = (1 + test_df['benchmark_return'] / 100).cumprod()

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

strategy_test_metrics = calculate_performance_metrics(test_df['strategy_return'], test_df['cumulative_strategy'])
benchmark_test_metrics = calculate_performance_metrics(test_df['benchmark_return'], test_df['cumulative_benchmark'])

oos_summary_table = pd.DataFrame([strategy_test_metrics, benchmark_test_metrics], index=[
    'OOS Sector Rotation Strategy (2016-Present)',
    'OOS Equal-Weight Benchmark (2016-Present)'
])

print("\n==========================================================================================")
print(f"        OUT-OF-SAMPLE (TEST PERIOD ONLY) PERFORMANCE RESULTS ({test_years:.1f} YEARS)      ")
print("==========================================================================================")
print(oos_summary_table.to_string())
