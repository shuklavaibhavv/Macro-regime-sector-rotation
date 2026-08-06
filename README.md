# Macro Regime Classification & Sector Rotation Strategy

## Overview
A systematic framework that classifies the U.S. economy into four macro 
regimes (Expansion, Slowdown, Recovery, Contraction) using manufacturing 
activity and Treasury yield curve data, then tests whether regime-aware 
sector rotation can outperform a passive benchmark. Built with a strict 
train/test split and lagged signals to avoid lookahead bias and overfitting.

## Pipeline (run in this order)
1. `src/fetch_data.py` — pulls PMI proxy (Industrial Production YoY) and 
   10Y-2Y yield curve spread from FRED, 2005-present
2. `src/regime_classifier.py` — defines the regime classification function 
   (4-quadrant logic + rate-of-change crisis override)
3. `src/build_regime_history.py` — applies the classifier across full history
4. `src/verify_regimes.py` — validates classifications against the 2008 GFC 
   and 2020 COVID crash
5. `src/fetch_sector_data.py` — pulls monthly total-return data for 9 S&P 
   sector ETFs via yfinance
6. `src/regime_sector_analysis.py` — computes average return and win rate 
   per sector, per regime
7. `src/contraction_split_analysis.py` — splits Contraction periods into 
   early/late phases to test whether outperformance is anticipatory-rally 
   driven
8. `src/backtest_rotation.py` — full-sample backtest (has in-sample bias, 
   kept for comparison)
9. `src/backtest_rotation_oos.py` — proper out-of-sample backtest: allocation 
   rule derived only from 2005-2015, tested only on 2016-2026
10. `src/backtest_rotation_voltarget.py` — tests a volatility-targeting 
    risk overlay on the OOS backtest

## Key Findings
- Out-of-sample (2016-2026), the strategy delivered 17.95% CAGR vs. 13.26% 
  for a passive equal-weight benchmark — a persistent, non-overfit edge
- But NOT on a risk-adjusted basis: Sharpe ratio was 0.87 vs. 0.91 for the 
  benchmark — the edge is compensation for higher volatility (20.57% vs. 
  14.64%), not efficiency
- Sector outperformance during Contraction concentrates in the late, 
  anticipatory-rally phase (e.g. Financials: -0.60% early to +2.97% late, 
  avg. monthly return) — except Technology, strong in both halves
- A 15%-volatility-target overlay, intended to improve risk-adjusted returns, 
  instead reduced Sharpe to 0.68 and CAGR to 11.17% — the strategy's edge is 
  concentrated in exactly the high-volatility, rally-driven months the 
  overlay suppresses

## Known Limitations & Future Work
- The classifier still slightly under-catches very gradual PMI declines 
  (e.g. May-Aug 2008) that don't cross either the 1-month or 3-month 
  rate-of-change thresholds
- PMI is proxied via Industrial Production YoY growth, not the official 
  (proprietary) ISM Manufacturing PMI series
- The Slowdown regime has a thin sample (16 months across 21 years), 
  limiting confidence in its sector rankings
- Future work: an asymmetric risk overlay distinguishing rally-driven from 
  panic-driven volatility; sub-phase-aware allocation using real-time-
  knowable proxies for regime age

## How to Run
1. `python3 -m venv venv && source venv/bin/activate`
2. `pip install -r requirements.txt`
3. Run scripts 1-10 above in order from the `src/` directory
4. Outputs (CSVs) are saved to `data/`

## Requirements
See requirements.txt (pandas, numpy, yfinance, matplotlib)
