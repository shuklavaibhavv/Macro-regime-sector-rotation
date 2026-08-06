import pandas as pd

df = pd.read_csv("data/regime_history.csv")
df['date'] = pd.to_datetime(df['date'])

print("==========================================================================")
print("              MACRO REGIME SANITY CHECK: KNOWN CRISIS WINDOWS            ")
print("==========================================================================")

print("\n--- 1. 2008 FINANCIAL CRISIS (2007-06 to 2009-12) ---")
gfc_df = df[(df['date'] >= '2007-06-01') & (df['date'] <= '2009-12-01')]
print(gfc_df.to_string(index=False))

print("\n--- 2. 2020 COVID CRISIS (2020-01 to 2020-12) ---")
covid_df = df[(df['date'] >= '2020-01-01') & (df['date'] <= '2020-12-01')]
print(covid_df.to_string(index=False))
