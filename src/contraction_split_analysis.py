import pandas as pd
import math

print("--- Contraction Phase Split Analysis: Early vs Late Contraction ---")

df = pd.read_csv("data/regime_sector_merged.csv")
df['regime_run_id'] = (df['regime'] != df['regime'].shift()).cumsum()

contraction_df = df[df['regime'] == 'Contraction'].copy()
distinct_contraction_runs = contraction_df['regime_run_id'].unique()

print(f"Total distinct contraction periods (recession cycles) identified: {len(distinct_contraction_runs)}\n")

sub_phase_labels = []

for run_id in distinct_contraction_runs:
    run_rows = contraction_df[contraction_df['regime_run_id'] == run_id]
    n_months = len(run_rows)
    early_count = math.ceil(n_months / 2)
    late_count = n_months - early_count
    run_labels = ['Early Contraction'] * early_count + ['Late Contraction'] * late_count
    sub_phase_labels.extend(run_labels)

contraction_df['sub_phase'] = sub_phase_labels
sectors = ['XLF', 'XLK', 'XLE', 'XLV', 'XLY', 'XLP', 'XLI', 'XLU', 'XLB']

early_vs_late_returns = contraction_df.groupby('sub_phase')[sectors].mean().round(2)
phase_counts = contraction_df['sub_phase'].value_counts()

print("==========================================================================")
print("             CONTRACTION MONTH COUNTS (EARLY VS LATE)                    ")
print("==========================================================================")
for phase, count in phase_counts.items():
    print(f"  • {phase}: {count} months")

print("\n==========================================================================")
print("     AVERAGE MONTHLY SECTOR RETURN (%): EARLY vs LATE CONTRACTION         ")
print("==========================================================================")
print(early_vs_late_returns.to_string())
