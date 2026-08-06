import sys
import os

# Add src folder to module search path so python can import regime_classifier cleanly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from regime_classifier import classify_regime
import pandas as pd

print("--- Classifying Historical Economic Data into Macro Regimes (v3 with 3-Mo Trend) ---")

# Load macro dataset from data/ directory
df = pd.read_csv("data/macro_data.csv")

# 1-Month and 3-Month PMI changes
df['pmi_change'] = df['pmi'].diff().fillna(0)
df['pmi_change_3mo'] = df['pmi'].diff(periods=3).fillna(0)

# Apply classifier
df['regime'] = df.apply(
    lambda row: classify_regime(
        row['pmi'], 
        row['yield_curve_spread'], 
        row['pmi_change'], 
        row['pmi_change_3mo']
    ),
    axis=1
)

# Save results to data/ directory
df.to_csv("data/regime_history.csv", index=False)
print("\nSuccessfully processed macro data and saved updated results to 'data/regime_history.csv'!\n")

counts = df['regime'].value_counts()
percentages = (df['regime'].value_counts(normalize=True) * 100).round(2)

summary_df = pd.DataFrame({
    'Months Count': counts,
    'Percentage (%)': percentages
})

print("=== REVISED HISTORICAL REGIME BREAKDOWN (2005 - PRESENT) ===")
print(summary_df)
