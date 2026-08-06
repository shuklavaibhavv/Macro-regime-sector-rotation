import pandas as pd
import numpy as np

print("--- Backtesting Macro Regime Sector Rotation Strategy ---")

df = pd.read_csv("data/regime_sector_merged.csv")
df['regime_lagged'] = df['regime'].shift(1)
df = df.dropna(subset=['regime_lagged']).copy()

sector_allocation_map = {
    'Expansion': 'XLE',
    'Slowdown': 'XLB',
    'Recovery': 'XLK',
    'Contraction': 'XLK'
}

df['chosen_sector'] = df['regime_lagged'].map(sector_allocation_map)
df['strategy_return'] = df.apply(lambda row: row[row['chosen_sector']], axis=1)

sectors = ['XLF', 'XLK', 'XLE', 'XLV', 'XLY', 'XLP', 'XLI', 'XLU', 'XLB']
df['benchmark_return'] = df[sectors].mean(axis=1)

df['cumulative_strategy'] = (1 + df['strategy_return'] / 100).cumprod()
df['cumulative_benchmark'] = (1 + df['benchmark_return'] / 100).cumprod()

n_months = len(df)
total_years = n_months / 12.0

def calculate_performance_metrics(monthly_returns, cum_series):
    total_return = (cum_series.iloc[-1] - 1) * 100
    cagr = ((cum_series.iloc[-1]) ** (1 / total_years) - 1) * 100
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

strategy_metrics = calculate_performance_metrics(df['strategy_return'], df['cumulative_strategy'])
benchmark_metrics = calculate_performance_metrics(df['benchmark_return'], df['cumulative_benchmark'])

summary_table = pd.DataFrame([strategy_metrics, benchmark_metrics], index=[
    'Sector Rotation Strategy (Macro-Lagged)',
    'Equal-Weight Benchmark (Passive 9-Sector)'
])

output_cols = [
    'date', 'regime_lagged', 'chosen_sector', 
    'strategy_return', 'benchmark_return', 
    'cumulative_strategy', 'cumulative_benchmark'
]
df[output_cols].to_csv("data/backtest_results.csv", index=False)

print("Successfully executed backtest and saved detailed monthly results to 'data/backtest_results.csv'!\n")
print("==========================================================================================")
print(f"             BACKTEST PERFORMANCE COMPARISON ({total_years:.1f} YEARS: 2005 - PRESENT)          ")
print("==========================================================================================")
print(summary_table.to_string())
