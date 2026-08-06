# Define a reusable function named 'classify_regime' that accepts four inputs:
# 1. 'pmi': current PMI level number (typically 30-60)
# 2. 'yield_curve_spread': 10-Year minus 2-Year Treasury yield spread (can be negative)
# 3. 'pmi_change': 1-month change in PMI (current month minus previous month)
# 4. 'pmi_change_3mo': 3-month cumulative change in PMI (current month minus PMI from 3 months ago)
def classify_regime(pmi, yield_curve_spread, pmi_change, pmi_change_3mo):
    # RATE OF CHANGE & SUSTAINED TREND OVERRIDE RULE:
    # 1. pmi < 45: Severe absolute economic contraction.
    # 2. pmi_change < -3: Sharp 1-month economic shock (e.g. COVID lockdown or Lehman collapse).
    # 3. pmi_change_3mo < -4: Sustained multi-month economic deterioration (e.g. slow-motion decline in early 2008).
    #
    # WHY USE -4 FOR THE 3-MONTH THRESHOLD?
    # A single-month shock threshold of -3 catches sudden spikes. However, economies often decay gradually.
    # A -4 threshold over 3 months (~1.33 points drop per month) catches steady, grinding multi-month downturns
    # before PMI breaches 45, without triggering false alarms from minor month-to-month noise.
    if pmi < 45 or pmi_change < -3 or pmi_change_3mo < -4:
        return "Contraction"
    
    # STANDARD 4-QUADRANT LOGIC FOR STABLE / NORMAL CONDITIONS:
    # Check if PMI is greater than 50 (growing economy) AND yield curve spread is positive (healthy yield curve)
    elif pmi > 50 and yield_curve_spread > 0:
        # Both indicators suggest standard economic expansion
        return "Expansion"
    
    # Check if PMI is greater than 50 (growth) BUT yield curve spread is zero or negative (inverted yield curve warning)
    elif pmi > 50 and yield_curve_spread <= 0:
        # High PMI with an inverted curve signals late-cycle slowdown
        return "Slowdown"
    
    # Check if PMI is 50 or below (weak economy) BUT yield curve spread is positive (yield curve un-inverting / recovering)
    elif pmi <= 50 and yield_curve_spread > 0:
        # Positive yield spread with moderate PMI indicates early policy stimulus / economic recovery
        return "Recovery"
    
    # If none of the above conditions met, PMI <= 50 and yield spread <= 0 (weak economy + inverted curve)
    else:
        # Both indicators signal recessionary conditions
        return "Contraction"


# This block ensures test code only runs when executing this file directly
if __name__ == "__main__":
    print("--- Macro Regime Classifier Test (v3 with 3-Month Trend) ---")
    
    # Test Scenario: Early 2008 slow decay (PMI: 48.82, 1-mo change: -1.62, 3-mo change: -4.01, spread: 1.63%)
    res_2008 = classify_regime(48.82, 1.63, -1.62, -4.01)
    print(f"Scenario Early 2008 - Result: {res_2008}")
