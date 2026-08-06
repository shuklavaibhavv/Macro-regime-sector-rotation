import pandas as pd

print("--- Macro Regime & Sector Performance Analysis ---")

df_regime = pd.read_csv("data/regime_history.csv")
df_sector = pd.read_csv("data/sector_returns.csv")

merged_df = pd.merge(df_regime, df_sector, on="date", how="inner")
merged_df.to_csv("data/regime_sector_merged.csv", index=False)
print(f"Successfully merged datasets! Combined rows: {len(merged_df)}\n")

sectors = ['XLF', 'XLK', 'XLE', 'XLV', 'XLY', 'XLP', 'XLI', 'XLU', 'XLB']

avg_returns = merged_df.groupby('regime')[sectors].mean().round(2)
win_rates = merged_df.groupby('regime')[sectors].apply(lambda g: (g > 0).mean() * 100).round(2)

avg_returns.to_csv("data/regime_sector_summary.csv")

print("==========================================================================")
print("             AVERAGE MONTHLY SECTOR RETURN (%) BY MACRO REGIME            ")
print("==========================================================================")
print(avg_returns.to_string())

print("\n==========================================================================")
print("                 SECTOR WIN RATE (%) BY MACRO REGIME                      ")
print("==========================================================================")
print(win_rates.to_string())

print("\n==========================================================================")
print("                 BEST AND WORST PERFORMING SECTOR PER REGIME              ")
print("==========================================================================")

for regime in avg_returns.index:
    row = avg_returns.loc[regime]
    best_sector = row.idxmax()
    best_return = row.max()
    worst_sector = row.idxmin()
    worst_return = row.min()
    
    print(f"\n[{regime.upper()} REGIME]")
    print(f"  🏆 Best Sector:  {best_sector} ({best_return}% average monthly return)")
    print(f"  ⚠️ Worst Sector: {worst_sector} ({worst_return}% average monthly return)")
