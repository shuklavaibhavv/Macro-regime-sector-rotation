"""
Macro Regime Sector Rotation — One-Click Dashboard
===================================================
Run this single file to:
  1. Execute the full 10-step data pipeline (fetch → classify → backtest)
  2. Launch a local web dashboard (http://127.0.0.1:8050) displaying every
     CSV output in rich, styled tables with charts.

Usage:
    python run_dashboard.py
"""

import subprocess, sys, os, math, webbrowser, threading, json
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

# ---------------------------------------------------------------------------
# 0.  RESOLVE PATHS
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
SRC  = ROOT / "src"
DATA = ROOT / "data"
os.chdir(ROOT)                       # all src/ scripts expect CWD = project root

# ---------------------------------------------------------------------------
# 1.  RUN THE FULL PIPELINE
# ---------------------------------------------------------------------------
SCRIPTS = [
    "fetch_data.py",
    "regime_classifier.py",
    "build_regime_history.py",
    "verify_regimes.py",
    "fetch_sector_data.py",
    "regime_sector_analysis.py",
    "contraction_split_analysis.py",
    "backtest_rotation.py",
    "backtest_rotation_oos.py",
    "backtest_rotation_voltarget.py",
]

print("=" * 70)
print("  MACRO REGIME SECTOR ROTATION — FULL PIPELINE")
print("=" * 70)

for i, script in enumerate(SCRIPTS, 1):
    path = SRC / script
    print(f"\n{'─'*70}")
    print(f"  [{i}/{len(SCRIPTS)}] Running {script} …")
    print(f"{'─'*70}")
    result = subprocess.run(
        [sys.executable, str(path)],
        cwd=str(ROOT),
    )
    if result.returncode != 0:
        print(f"\n⚠  {script} exited with code {result.returncode}")
        sys.exit(1)

print(f"\n{'='*70}")
print("  ✅  ALL PIPELINE SCRIPTS COMPLETED SUCCESSFULLY")
print(f"{'='*70}\n")

# ---------------------------------------------------------------------------
# 2.  READ CSV DATA AND CONVERT TO JSON FOR THE DASHBOARD
# ---------------------------------------------------------------------------
import pandas as pd
import numpy as np

def df_to_json(df):
    """Convert a DataFrame to a list of dicts, handling NaN."""
    return json.loads(df.fillna("").to_json(orient="records"))

dashboard_data = {}

# --- Macro Data ---
macro_df = pd.read_csv(DATA / "macro_data.csv")
dashboard_data["macro_data"] = df_to_json(macro_df)

# --- Regime History ---
regime_df = pd.read_csv(DATA / "regime_history.csv")
dashboard_data["regime_history"] = df_to_json(regime_df)

# Regime breakdown counts
regime_counts = regime_df["regime"].value_counts().to_dict()
regime_pcts = (regime_df["regime"].value_counts(normalize=True) * 100).round(2).to_dict()
dashboard_data["regime_breakdown"] = {
    r: {"count": regime_counts[r], "pct": regime_pcts[r]} for r in regime_counts
}

# --- Sector Returns ---
sector_df = pd.read_csv(DATA / "sector_returns.csv")
dashboard_data["sector_returns"] = df_to_json(sector_df)

# --- Regime Sector Merged ---
merged_df = pd.read_csv(DATA / "regime_sector_merged.csv")
dashboard_data["regime_sector_merged"] = df_to_json(merged_df)

# --- Regime Sector Summary ---
summary_df = pd.read_csv(DATA / "regime_sector_summary.csv")
dashboard_data["regime_sector_summary"] = df_to_json(summary_df)

# --- Backtest Results ---
backtest_df = pd.read_csv(DATA / "backtest_results.csv")
dashboard_data["backtest_results"] = df_to_json(backtest_df)

# --- Compute performance metrics for display cards ---
sectors = ['XLF', 'XLK', 'XLE', 'XLV', 'XLY', 'XLP', 'XLI', 'XLU', 'XLB']

def calc_metrics(monthly_returns, cum_series, years):
    total_return = (cum_series.iloc[-1] - 1) * 100
    cagr = ((cum_series.iloc[-1]) ** (1 / years) - 1) * 100
    ann_vol = monthly_returns.std() * np.sqrt(12)
    sharpe = cagr / ann_vol if ann_vol != 0 else 0
    peak = cum_series.cummax()
    dd = ((cum_series - peak) / peak).min() * 100
    return {
        "total_return": round(total_return, 2),
        "cagr": round(cagr, 2),
        "ann_vol": round(ann_vol, 2),
        "sharpe": round(sharpe, 2),
        "max_dd": round(dd, 2),
    }

# Full-sample backtest metrics
bt = backtest_df.copy()
bt_years = len(bt) / 12.0
bt["cum_s"] = (1 + bt["strategy_return"] / 100).cumprod()
bt["cum_b"] = (1 + bt["benchmark_return"] / 100).cumprod()
dashboard_data["full_backtest_metrics"] = {
    "strategy": calc_metrics(bt["strategy_return"], bt["cum_s"], bt_years),
    "benchmark": calc_metrics(bt["benchmark_return"], bt["cum_b"], bt_years),
}

# OOS backtest metrics
oos_merged = pd.read_csv(DATA / "regime_sector_merged.csv")
oos_merged["regime_lagged"] = oos_merged["regime"].shift(1)
oos_merged = oos_merged.dropna(subset=["regime_lagged"]).copy()
train_oos = oos_merged[oos_merged["date"] < "2016-01-01"]
test_oos = oos_merged[oos_merged["date"] >= "2016-01-01"].copy()
train_avg = train_oos.groupby("regime_lagged")[sectors].mean()
alloc_map = {r: train_avg.loc[r].idxmax() for r in train_avg.index}
test_oos["chosen_sector"] = test_oos["regime_lagged"].map(alloc_map)
test_oos["strategy_return"] = test_oos.apply(lambda row: row[row["chosen_sector"]], axis=1)
test_oos["benchmark_return"] = test_oos[sectors].mean(axis=1)
test_oos["cum_s"] = (1 + test_oos["strategy_return"] / 100).cumprod()
test_oos["cum_b"] = (1 + test_oos["benchmark_return"] / 100).cumprod()
oos_years = len(test_oos) / 12.0
dashboard_data["oos_metrics"] = {
    "strategy": calc_metrics(test_oos["strategy_return"], test_oos["cum_s"], oos_years),
    "benchmark": calc_metrics(test_oos["benchmark_return"], test_oos["cum_b"], oos_years),
    "allocation_map": alloc_map,
}

# Contraction early/late
con_df = merged_df.copy()
con_df["regime_run_id"] = (con_df["regime"] != con_df["regime"].shift()).cumsum()
con_only = con_df[con_df["regime"] == "Contraction"].copy()
sub_labels = []
for run_id in con_only["regime_run_id"].unique():
    rows = con_only[con_only["regime_run_id"] == run_id]
    n = len(rows)
    early = math.ceil(n / 2)
    sub_labels.extend(["Early Contraction"] * early + ["Late Contraction"] * (n - early))
con_only["sub_phase"] = sub_labels
early_late = con_only.groupby("sub_phase")[sectors].mean().round(2)
dashboard_data["contraction_split"] = {
    phase: early_late.loc[phase].to_dict() for phase in early_late.index
}

# Serialize — replace any lingering NaN / Inf with None
import math as _math

def _sanitize(obj):
    if isinstance(obj, float) and (_math.isnan(obj) or _math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj

data_json = json.dumps(_sanitize(dashboard_data), default=str)

# ---------------------------------------------------------------------------
# 3.  GENERATE THE DASHBOARD HTML
# ---------------------------------------------------------------------------
HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Macro Regime Sector Rotation Dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet">
<style>
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

:root {{
  --bg-primary: #0a0e1a;
  --bg-secondary: #111827;
  --bg-card: #1a1f35;
  --bg-card-hover: #222845;
  --border: rgba(255,255,255,0.06);
  --text-primary: #e8ecf4;
  --text-secondary: #8b95b0;
  --text-muted: #5a6380;
  --accent-blue: #3b82f6;
  --accent-purple: #8b5cf6;
  --accent-green: #10b981;
  --accent-red: #ef4444;
  --accent-amber: #f59e0b;
  --accent-cyan: #06b6d4;
  --gradient-1: linear-gradient(135deg, #3b82f6, #8b5cf6);
  --gradient-2: linear-gradient(135deg, #10b981, #06b6d4);
  --gradient-3: linear-gradient(135deg, #f59e0b, #ef4444);
  --gradient-4: linear-gradient(135deg, #ec4899, #8b5cf6);
  --shadow: 0 4px 24px rgba(0,0,0,0.3);
  --shadow-lg: 0 8px 40px rgba(0,0,0,0.4);
}}

body {{
  font-family: 'Inter', -apple-system, sans-serif;
  background: var(--bg-primary);
  color: var(--text-primary);
  line-height: 1.6;
  min-height: 100vh;
}}

/* --- HEADER --- */
.header {{
  background: linear-gradient(135deg, rgba(59,130,246,0.12), rgba(139,92,246,0.12));
  border-bottom: 1px solid var(--border);
  padding: 2rem 2rem 1.5rem;
  position: sticky; top: 0; z-index: 100;
  backdrop-filter: blur(20px);
}}
.header h1 {{
  font-size: 1.8rem; font-weight: 800;
  background: var(--gradient-1); -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  letter-spacing: -0.5px;
}}
.header p {{ color: var(--text-secondary); font-size: 0.85rem; margin-top: 0.3rem; }}

/* --- NAV TABS --- */
.nav {{ display: flex; gap: 0.5rem; padding: 1rem 2rem; background: var(--bg-secondary); border-bottom: 1px solid var(--border); flex-wrap: wrap; }}
.nav button {{
  padding: 0.5rem 1.2rem; border: 1px solid var(--border); border-radius: 8px;
  background: transparent; color: var(--text-secondary); font-size: 0.8rem;
  font-weight: 500; cursor: pointer; transition: all 0.25s;
  font-family: inherit;
}}
.nav button:hover {{ background: rgba(59,130,246,0.1); color: var(--text-primary); border-color: var(--accent-blue); }}
.nav button.active {{
  background: var(--gradient-1); color: white; border-color: transparent;
  box-shadow: 0 2px 12px rgba(59,130,246,0.3);
}}

/* --- MAIN --- */
.main {{ padding: 1.5rem 2rem 4rem; max-width: 1500px; margin: 0 auto; }}
.section {{ display: none; animation: fadeIn 0.4s ease; }}
.section.active {{ display: block; }}
@keyframes fadeIn {{ from {{ opacity: 0; transform: translateY(12px); }} to {{ opacity: 1; transform: translateY(0); }} }}

.section-title {{
  font-size: 1.3rem; font-weight: 700; margin-bottom: 1.2rem;
  display: flex; align-items: center; gap: 0.6rem;
}}
.section-title .dot {{ width: 10px; height: 10px; border-radius: 50%; }}

/* --- METRIC CARDS --- */
.metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 1.5rem; }}
.metric-card {{
  background: var(--bg-card); border: 1px solid var(--border); border-radius: 12px;
  padding: 1.2rem; position: relative; overflow: hidden; transition: transform 0.2s, box-shadow 0.2s;
}}
.metric-card:hover {{ transform: translateY(-2px); box-shadow: var(--shadow); }}
.metric-card .label {{ font-size: 0.7rem; text-transform: uppercase; letter-spacing: 1px; color: var(--text-muted); font-weight: 600; }}
.metric-card .value {{ font-size: 1.6rem; font-weight: 800; margin-top: 0.3rem; }}
.metric-card .sub {{ font-size: 0.75rem; color: var(--text-secondary); margin-top: 0.2rem; }}
.metric-card::before {{
  content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
}}
.metric-card.blue::before {{ background: var(--gradient-1); }}
.metric-card.green::before {{ background: var(--gradient-2); }}
.metric-card.red::before {{ background: var(--gradient-3); }}
.metric-card.purple::before {{ background: var(--gradient-4); }}

/* --- REGIME BADGES --- */
.regime-badge {{
  display: inline-block; padding: 0.15rem 0.6rem; border-radius: 20px;
  font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;
}}
.regime-Expansion {{ background: rgba(16,185,129,0.15); color: #34d399; }}
.regime-Slowdown {{ background: rgba(245,158,11,0.15); color: #fbbf24; }}
.regime-Recovery {{ background: rgba(59,130,246,0.15); color: #60a5fa; }}
.regime-Contraction {{ background: rgba(239,68,68,0.15); color: #f87171; }}

/* --- TABLES --- */
.table-wrap {{
  background: var(--bg-card); border: 1px solid var(--border); border-radius: 12px;
  overflow: hidden; margin-bottom: 1.5rem;
}}
.table-title {{
  padding: 1rem 1.2rem; font-weight: 600; font-size: 0.9rem;
  border-bottom: 1px solid var(--border);
  background: rgba(255,255,255,0.02);
}}
.table-scroll {{ overflow-x: auto; max-height: 500px; overflow-y: auto; }}
table {{
  width: 100%; border-collapse: collapse; font-size: 0.78rem;
}}
thead {{ position: sticky; top: 0; z-index: 2; }}
th {{
  padding: 0.7rem 1rem; text-align: left; font-weight: 600; font-size: 0.7rem;
  text-transform: uppercase; letter-spacing: 0.8px; color: var(--text-muted);
  background: var(--bg-secondary); border-bottom: 1px solid var(--border);
  white-space: nowrap;
}}
td {{
  padding: 0.55rem 1rem; border-bottom: 1px solid var(--border);
  color: var(--text-secondary); white-space: nowrap;
}}
tr:hover td {{ background: rgba(59,130,246,0.04); color: var(--text-primary); }}
.positive {{ color: var(--accent-green); }}
.negative {{ color: var(--accent-red); }}

/* --- ALLOCATION MAP --- */
.alloc-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 0.8rem; margin-bottom: 1.5rem; }}
.alloc-card {{
  background: var(--bg-card); border: 1px solid var(--border); border-radius: 10px;
  padding: 1rem; text-align: center; transition: transform 0.2s;
}}
.alloc-card:hover {{ transform: translateY(-2px); }}
.alloc-card .regime-name {{ font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1px; }}
.alloc-card .arrow {{ font-size: 1.2rem; margin: 0.3rem 0; color: var(--text-muted); }}
.alloc-card .sector-name {{ font-size: 1.1rem; font-weight: 700; color: var(--accent-cyan); }}

/* --- CHART AREA --- */
.chart-container {{
  background: var(--bg-card); border: 1px solid var(--border); border-radius: 12px;
  padding: 1.5rem; margin-bottom: 1.5rem; position: relative;
}}
.chart-container canvas {{ width: 100% !important; height: 350px !important; }}

/* --- DUAL GRID --- */
.dual-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.2rem; }}
@media (max-width: 900px) {{ .dual-grid {{ grid-template-columns: 1fr; }} }}

/* --- SCROLLBAR --- */
::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: var(--bg-secondary); }}
::-webkit-scrollbar-thumb {{ background: #333a55; border-radius: 3px; }}
::-webkit-scrollbar-thumb:hover {{ background: #444b6a; }}

/* --- SEARCH / FILTER --- */
.search-bar {{
  display: flex; gap: 0.6rem; margin-bottom: 1rem; flex-wrap: wrap; align-items: center;
}}
.search-bar input {{
  padding: 0.5rem 1rem; border: 1px solid var(--border); border-radius: 8px;
  background: var(--bg-secondary); color: var(--text-primary); font-size: 0.8rem;
  font-family: inherit; outline: none; min-width: 220px;
}}
.search-bar input:focus {{ border-color: var(--accent-blue); }}
.search-bar select {{
  padding: 0.5rem 0.8rem; border: 1px solid var(--border); border-radius: 8px;
  background: var(--bg-secondary); color: var(--text-primary); font-size: 0.8rem;
  font-family: inherit; outline: none; cursor: pointer;
}}
.row-count {{ font-size: 0.75rem; color: var(--text-muted); margin-left: auto; }}
</style>
</head>
<body>

<div class="header">
  <h1>📊 Macro Regime Sector Rotation</h1>
  <p>Full pipeline executed &amp; all data rendered — single-run dashboard</p>
</div>

<div class="nav" id="nav">
  <button class="active" data-tab="overview">Overview</button>
  <button data-tab="macro">Macro Data</button>
  <button data-tab="regimes">Regime History</button>
  <button data-tab="sectors">Sector Returns</button>
  <button data-tab="analysis">Regime × Sector</button>
  <button data-tab="contraction">Contraction Split</button>
  <button data-tab="backtest">Full Backtest</button>
  <button data-tab="oos">OOS Backtest</button>
</div>

<div class="main">

  <!-- ============= OVERVIEW ============= -->
  <div class="section active" id="tab-overview">
    <div class="section-title"><span class="dot" style="background:var(--gradient-1)"></span>Dashboard Overview</div>

    <div class="metric-grid" id="overview-metrics"></div>

    <div class="dual-grid">
      <div class="chart-container">
        <div style="font-weight:600;margin-bottom:0.8rem;font-size:0.9rem;">📈 Full-Sample Cumulative Returns</div>
        <canvas id="chart-full-cum"></canvas>
      </div>
      <div class="chart-container">
        <div style="font-weight:600;margin-bottom:0.8rem;font-size:0.9rem;">📈 Out-of-Sample Cumulative Returns (2016+)</div>
        <canvas id="chart-oos-cum"></canvas>
      </div>
    </div>

    <div class="dual-grid">
      <div class="chart-container">
        <div style="font-weight:600;margin-bottom:0.8rem;font-size:0.9rem;">🎯 Regime Distribution</div>
        <canvas id="chart-regime-pie"></canvas>
      </div>
      <div>
        <div class="section-title" style="margin-top:0.5rem;"><span class="dot" style="background:var(--gradient-2)"></span>OOS Allocation Map</div>
        <div class="alloc-grid" id="alloc-map"></div>
      </div>
    </div>
  </div>

  <!-- ============= MACRO DATA ============= -->
  <div class="section" id="tab-macro">
    <div class="section-title"><span class="dot" style="background:var(--accent-cyan)"></span>Macro Economic Data (FRED)</div>
    <div class="chart-container">
      <div style="font-weight:600;margin-bottom:0.8rem;font-size:0.9rem;">PMI Proxy &amp; Yield Curve Spread Over Time</div>
      <canvas id="chart-macro"></canvas>
    </div>
    <div class="table-wrap">
      <div class="table-title">macro_data.csv</div>
      <div class="search-bar" style="padding:0.8rem 1.2rem 0;">
        <input type="text" placeholder="Search dates..." id="search-macro">
        <span class="row-count" id="count-macro"></span>
      </div>
      <div class="table-scroll" id="table-macro"></div>
    </div>
  </div>

  <!-- ============= REGIME HISTORY ============= -->
  <div class="section" id="tab-regimes">
    <div class="section-title"><span class="dot" style="background:var(--accent-green)"></span>Regime Classification History</div>
    <div class="metric-grid" id="regime-counts"></div>
    <div class="table-wrap">
      <div class="table-title">regime_history.csv</div>
      <div class="search-bar" style="padding:0.8rem 1.2rem 0;">
        <input type="text" placeholder="Search..." id="search-regime">
        <select id="filter-regime">
          <option value="">All Regimes</option>
          <option value="Expansion">Expansion</option>
          <option value="Slowdown">Slowdown</option>
          <option value="Recovery">Recovery</option>
          <option value="Contraction">Contraction</option>
        </select>
        <span class="row-count" id="count-regime"></span>
      </div>
      <div class="table-scroll" id="table-regime"></div>
    </div>
  </div>

  <!-- ============= SECTOR RETURNS ============= -->
  <div class="section" id="tab-sectors">
    <div class="section-title"><span class="dot" style="background:var(--accent-amber)"></span>Monthly Sector ETF Returns (%)</div>
    <div class="chart-container">
      <div style="font-weight:600;margin-bottom:0.8rem;font-size:0.9rem;">Sector Returns Heatmap (last 24 months)</div>
      <canvas id="chart-sector-heat"></canvas>
    </div>
    <div class="table-wrap">
      <div class="table-title">sector_returns.csv</div>
      <div class="search-bar" style="padding:0.8rem 1.2rem 0;">
        <input type="text" placeholder="Search dates..." id="search-sector">
        <span class="row-count" id="count-sector"></span>
      </div>
      <div class="table-scroll" id="table-sector"></div>
    </div>
  </div>

  <!-- ============= REGIME × SECTOR ============= -->
  <div class="section" id="tab-analysis">
    <div class="section-title"><span class="dot" style="background:var(--accent-purple)"></span>Average Sector Return by Regime</div>
    <div class="chart-container">
      <div style="font-weight:600;margin-bottom:0.8rem;font-size:0.9rem;">Avg Monthly Return (%) by Regime × Sector</div>
      <canvas id="chart-regime-sector"></canvas>
    </div>
    <div class="table-wrap">
      <div class="table-title">regime_sector_summary.csv — Average Monthly Returns (%)</div>
      <div class="table-scroll" id="table-summary"></div>
    </div>
  </div>

  <!-- ============= CONTRACTION SPLIT ============= -->
  <div class="section" id="tab-contraction">
    <div class="section-title"><span class="dot" style="background:var(--accent-red)"></span>Contraction Phase Split: Early vs Late</div>
    <div class="chart-container">
      <div style="font-weight:600;margin-bottom:0.8rem;font-size:0.9rem;">Early vs Late Contraction — Avg Monthly Return (%)</div>
      <canvas id="chart-contraction"></canvas>
    </div>
    <div class="table-wrap">
      <div class="table-title">Early vs Late Contraction Average Returns</div>
      <div class="table-scroll" id="table-contraction"></div>
    </div>
  </div>

  <!-- ============= FULL BACKTEST ============= -->
  <div class="section" id="tab-backtest">
    <div class="section-title"><span class="dot" style="background:var(--gradient-1)"></span>Full-Sample Backtest (2005–Present)</div>
    <div class="metric-grid" id="bt-metrics"></div>
    <div class="chart-container">
      <div style="font-weight:600;margin-bottom:0.8rem;font-size:0.9rem;">Cumulative Returns — Strategy vs Benchmark</div>
      <canvas id="chart-bt-full"></canvas>
    </div>
    <div class="table-wrap">
      <div class="table-title">backtest_results.csv</div>
      <div class="search-bar" style="padding:0.8rem 1.2rem 0;">
        <input type="text" placeholder="Search..." id="search-bt">
        <select id="filter-bt-regime">
          <option value="">All Regimes</option>
          <option value="Expansion">Expansion</option>
          <option value="Slowdown">Slowdown</option>
          <option value="Recovery">Recovery</option>
          <option value="Contraction">Contraction</option>
        </select>
        <span class="row-count" id="count-bt"></span>
      </div>
      <div class="table-scroll" id="table-bt"></div>
    </div>
  </div>

  <!-- ============= OOS BACKTEST ============= -->
  <div class="section" id="tab-oos">
    <div class="section-title"><span class="dot" style="background:var(--gradient-2)"></span>Out-of-Sample Backtest (2016–Present)</div>
    <div class="metric-grid" id="oos-metrics-cards"></div>
    <div class="chart-container">
      <div style="font-weight:600;margin-bottom:0.8rem;font-size:0.9rem;">OOS Cumulative Returns — Strategy vs Benchmark</div>
      <canvas id="chart-oos-full"></canvas>
    </div>
  </div>

</div>

<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<script>
// ─── DATA ───
const D = {data_json};

// ─── TAB SWITCHING ───
document.querySelectorAll('.nav button').forEach(btn => {{
  btn.addEventListener('click', () => {{
    document.querySelectorAll('.nav button').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
    document.getElementById('tab-' + btn.dataset.tab).classList.add('active');
  }});
}});

// ─── HELPERS ───
function colorVal(v) {{
  if (v === null || v === undefined || v === '') return '';
  const n = parseFloat(v);
  if (isNaN(n)) return v;
  if (n > 0) return `<span class="positive">+${{n.toFixed(2)}}</span>`;
  if (n < 0) return `<span class="negative">${{n.toFixed(2)}}</span>`;
  return n.toFixed(2);
}}

function regimeBadge(r) {{
  return `<span class="regime-badge regime-${{r}}">${{r}}</span>`;
}}

function buildTable(containerId, rows, columns, opts = {{}}) {{
  if (!rows || rows.length === 0) {{ document.getElementById(containerId).innerHTML = '<p style="padding:1rem;color:var(--text-muted)">No data</p>'; return; }}
  const colorCols = opts.colorCols || [];
  const regimeCol = opts.regimeCol || null;
  let html = '<table><thead><tr>';
  columns.forEach(c => html += `<th>${{c}}</th>`);
  html += '</tr></thead><tbody>';
  rows.forEach(row => {{
    html += '<tr>';
    columns.forEach(c => {{
      let v = row[c];
      if (c === regimeCol && v) html += `<td>${{regimeBadge(v)}}</td>`;
      else if (colorCols.includes(c)) html += `<td>${{colorVal(v)}}</td>`;
      else html += `<td>${{v !== null && v !== undefined ? v : ''}}</td>`;
    }});
    html += '</tr>';
  }});
  html += '</tbody></table>';
  document.getElementById(containerId).innerHTML = html;
}}

function setupSearch(inputId, countId, data, columns, containerId, opts = {{}}) {{
  const input = document.getElementById(inputId);
  const filterSel = opts.filterSelect ? document.getElementById(opts.filterSelect) : null;
  const filterKey = opts.filterKey || null;
  const update = () => {{
    const q = input.value.toLowerCase();
    const fv = filterSel ? filterSel.value : '';
    let filtered = data.filter(row => {{
      const matchQ = !q || columns.some(c => String(row[c] || '').toLowerCase().includes(q));
      const matchF = !fv || (filterKey && row[filterKey] === fv);
      return matchQ && matchF;
    }});
    buildTable(containerId, filtered, columns, opts);
    document.getElementById(countId).textContent = `${{filtered.length}} / ${{data.length}} rows`;
  }};
  input.addEventListener('input', update);
  if (filterSel) filterSel.addEventListener('change', update);
  update();
}}

// ─── OVERVIEW ───
(function() {{
  const fm = D.full_backtest_metrics;
  const om = D.oos_metrics;
  const cards = [
    {{ label: 'Full CAGR (Strategy)', value: fm.strategy.cagr + '%', sub: 'vs ' + fm.benchmark.cagr + '% benchmark', cls: 'blue' }},
    {{ label: 'Full Sharpe', value: fm.strategy.sharpe, sub: 'vs ' + fm.benchmark.sharpe + ' benchmark', cls: 'purple' }},
    {{ label: 'OOS CAGR (Strategy)', value: om.strategy.cagr + '%', sub: '2016–present, vs ' + om.benchmark.cagr + '%', cls: 'green' }},
    {{ label: 'OOS Sharpe', value: om.strategy.sharpe, sub: 'vs ' + om.benchmark.sharpe + ' benchmark', cls: 'purple' }},
    {{ label: 'Full Max Drawdown', value: fm.strategy.max_dd + '%', sub: 'Strategy worst peak-to-trough', cls: 'red' }},
    {{ label: 'OOS Max Drawdown', value: om.strategy.max_dd + '%', sub: 'Strategy 2016+', cls: 'red' }},
  ];
  let html = '';
  cards.forEach(c => {{
    html += `<div class="metric-card ${{c.cls}}"><div class="label">${{c.label}}</div><div class="value">${{c.value}}</div><div class="sub">${{c.sub}}</div></div>`;
  }});
  document.getElementById('overview-metrics').innerHTML = html;

  // Allocation map
  let allocHtml = '';
  for (const [regime, sector] of Object.entries(om.allocation_map)) {{
    allocHtml += `<div class="alloc-card"><div class="regime-name">${{regime}}</div><div class="arrow">↓</div><div class="sector-name">${{sector}}</div></div>`;
  }}
  document.getElementById('alloc-map').innerHTML = allocHtml;
}})();

// ─── CHARTS ───
const chartColors = {{
  strategy: '#3b82f6',
  benchmark: '#8b5cf6',
  strategyFill: 'rgba(59,130,246,0.08)',
  benchmarkFill: 'rgba(139,92,246,0.08)',
}};

const defaultOpts = {{
  responsive: true,
  maintainAspectRatio: false,
  plugins: {{
    legend: {{ labels: {{ color: '#8b95b0', font: {{ size: 11, family: 'Inter' }} }} }},
  }},
  scales: {{
    x: {{ ticks: {{ color: '#5a6380', font: {{ size: 10 }}, maxTicksLimit: 15 }}, grid: {{ color: 'rgba(255,255,255,0.04)' }} }},
    y: {{ ticks: {{ color: '#5a6380', font: {{ size: 10 }} }}, grid: {{ color: 'rgba(255,255,255,0.04)' }} }},
  }},
}};

// Full cumulative chart
(function() {{
  const bt = D.backtest_results;
  const labels = bt.map(r => r.date);
  let cumS = [], cumB = [], cs = 1, cb = 1;
  bt.forEach(r => {{
    cs *= (1 + r.strategy_return / 100);
    cb *= (1 + r.benchmark_return / 100);
    cumS.push(cs); cumB.push(cb);
  }});
  new Chart(document.getElementById('chart-full-cum'), {{
    type: 'line',
    data: {{
      labels,
      datasets: [
        {{ label: 'Strategy', data: cumS, borderColor: chartColors.strategy, backgroundColor: chartColors.strategyFill, fill: true, borderWidth: 2, pointRadius: 0, tension: 0.3 }},
        {{ label: 'Benchmark', data: cumB, borderColor: chartColors.benchmark, backgroundColor: chartColors.benchmarkFill, fill: true, borderWidth: 2, pointRadius: 0, tension: 0.3 }},
      ]
    }},
    options: defaultOpts
  }});
}})();

// OOS cumulative chart (overview)
(function() {{
  const bt = D.backtest_results.filter(r => r.date >= '2016-01-01');
  const labels = bt.map(r => r.date);
  let cumS = [], cumB = [], cs = 1, cb = 1;
  bt.forEach(r => {{
    cs *= (1 + r.strategy_return / 100);
    cb *= (1 + r.benchmark_return / 100);
    cumS.push(cs); cumB.push(cb);
  }});
  new Chart(document.getElementById('chart-oos-cum'), {{
    type: 'line',
    data: {{
      labels,
      datasets: [
        {{ label: 'Strategy (OOS)', data: cumS, borderColor: '#10b981', backgroundColor: 'rgba(16,185,129,0.08)', fill: true, borderWidth: 2, pointRadius: 0, tension: 0.3 }},
        {{ label: 'Benchmark', data: cumB, borderColor: '#06b6d4', backgroundColor: 'rgba(6,182,212,0.08)', fill: true, borderWidth: 2, pointRadius: 0, tension: 0.3 }},
      ]
    }},
    options: defaultOpts
  }});
}})();

// Regime pie
(function() {{
  const rb = D.regime_breakdown;
  const regimes = Object.keys(rb);
  const colors = {{ Expansion: '#10b981', Slowdown: '#f59e0b', Recovery: '#3b82f6', Contraction: '#ef4444' }};
  new Chart(document.getElementById('chart-regime-pie'), {{
    type: 'doughnut',
    data: {{
      labels: regimes,
      datasets: [{{ data: regimes.map(r => rb[r].count), backgroundColor: regimes.map(r => colors[r] || '#666'), borderWidth: 0 }}],
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{ legend: {{ position: 'right', labels: {{ color: '#8b95b0', padding: 15, font: {{ size: 12, family: 'Inter' }} }} }} }},
      cutout: '60%',
    }}
  }});
}})();

// ─── MACRO TAB ───
(function() {{
  const md = D.macro_data;
  const labels = md.map(r => r.date);
  new Chart(document.getElementById('chart-macro'), {{
    type: 'line',
    data: {{
      labels,
      datasets: [
        {{ label: 'PMI Proxy', data: md.map(r => r.pmi), borderColor: '#3b82f6', borderWidth: 2, pointRadius: 0, tension: 0.3, yAxisID: 'y' }},
        {{ label: 'Yield Curve Spread', data: md.map(r => r.yield_curve_spread), borderColor: '#f59e0b', borderWidth: 2, pointRadius: 0, tension: 0.3, yAxisID: 'y1' }},
      ]
    }},
    options: {{
      ...defaultOpts,
      scales: {{
        ...defaultOpts.scales,
        y: {{ ...defaultOpts.scales.y, position: 'left', title: {{ display: true, text: 'PMI Proxy', color: '#5a6380' }} }},
        y1: {{ ...defaultOpts.scales.y, position: 'right', title: {{ display: true, text: 'Yield Spread (%)', color: '#5a6380' }}, grid: {{ drawOnChartArea: false }} }},
      }}
    }}
  }});
  setupSearch('search-macro', 'count-macro', md, ['date','pmi','yield_curve_spread'], 'table-macro', {{ colorCols: ['pmi','yield_curve_spread'] }});
}})();

// ─── REGIME HISTORY TAB ───
(function() {{
  const rb = D.regime_breakdown;
  let html = '';
  const colors = {{ Expansion: 'green', Slowdown: 'purple', Recovery: 'blue', Contraction: 'red' }};
  for (const [r, info] of Object.entries(rb)) {{
    html += `<div class="metric-card ${{colors[r] || 'blue'}}"><div class="label">${{r}}</div><div class="value">${{info.count}} <span style="font-size:0.8rem;font-weight:400">months</span></div><div class="sub">${{info.pct}}% of total</div></div>`;
  }}
  document.getElementById('regime-counts').innerHTML = html;

  const cols = Object.keys(D.regime_history[0] || {{}});
  setupSearch('search-regime', 'count-regime', D.regime_history, cols, 'table-regime', {{
    regimeCol: 'regime', colorCols: ['pmi','yield_curve_spread','pmi_change','pmi_change_3mo'],
    filterSelect: 'filter-regime', filterKey: 'regime',
  }});
}})();

// ─── SECTOR RETURNS TAB ───
(function() {{
  const sr = D.sector_returns;
  const sectors = ['XLF','XLK','XLE','XLV','XLY','XLP','XLI','XLU','XLB'];

  // Heatmap as grouped bar (last 24 months)
  const last24 = sr.slice(-24);
  const sectorColors = ['#3b82f6','#8b5cf6','#10b981','#ef4444','#f59e0b','#06b6d4','#ec4899','#84cc16','#f97316'];
  new Chart(document.getElementById('chart-sector-heat'), {{
    type: 'bar',
    data: {{
      labels: last24.map(r => r.date),
      datasets: sectors.map((s, i) => ({{
        label: s, data: last24.map(r => r[s]),
        backgroundColor: sectorColors[i] + '99', borderColor: sectorColors[i], borderWidth: 1,
      }})),
    }},
    options: {{ ...defaultOpts, plugins: {{ ...defaultOpts.plugins, legend: {{ ...defaultOpts.plugins.legend, position: 'top' }} }} }}
  }});

  setupSearch('search-sector', 'count-sector', sr, ['date', ...sectors], 'table-sector', {{ colorCols: sectors }});
}})();

// ─── REGIME × SECTOR ANALYSIS ───
(function() {{
  const summary = D.regime_sector_summary;
  const sectors = ['XLF','XLK','XLE','XLV','XLY','XLP','XLI','XLU','XLB'];
  const regimeCol = summary[0] ? Object.keys(summary[0])[0] : 'regime';

  const regimeColors = {{ Expansion: '#10b981', Slowdown: '#f59e0b', Recovery: '#3b82f6', Contraction: '#ef4444' }};
  new Chart(document.getElementById('chart-regime-sector'), {{
    type: 'bar',
    data: {{
      labels: sectors,
      datasets: summary.map(row => ({{
        label: row[regimeCol],
        data: sectors.map(s => row[s] || 0),
        backgroundColor: (regimeColors[row[regimeCol]] || '#666') + 'cc',
        borderColor: regimeColors[row[regimeCol]] || '#666',
        borderWidth: 1,
      }})),
    }},
    options: {{ ...defaultOpts, plugins: {{ ...defaultOpts.plugins }} }}
  }});

  const allCols = Object.keys(summary[0] || {{}});
  buildTable('table-summary', summary, allCols, {{ colorCols: sectors, regimeCol }});
}})();

// ─── CONTRACTION SPLIT ───
(function() {{
  const cs = D.contraction_split;
  const sectors = ['XLF','XLK','XLE','XLV','XLY','XLP','XLI','XLU','XLB'];
  const phases = Object.keys(cs);

  new Chart(document.getElementById('chart-contraction'), {{
    type: 'bar',
    data: {{
      labels: sectors,
      datasets: phases.map((p, i) => ({{
        label: p,
        data: sectors.map(s => cs[p][s] || 0),
        backgroundColor: i === 0 ? 'rgba(239,68,68,0.6)' : 'rgba(16,185,129,0.6)',
        borderColor: i === 0 ? '#ef4444' : '#10b981',
        borderWidth: 1,
      }})),
    }},
    options: defaultOpts
  }});

  // Table
  const rows = phases.map(p => ({{ Phase: p, ...cs[p] }}));
  buildTable('table-contraction', rows, ['Phase', ...sectors], {{ colorCols: sectors }});
}})();

// ─── FULL BACKTEST TAB ───
(function() {{
  const fm = D.full_backtest_metrics;
  const labels = ['CAGR (%)', 'Sharpe', 'Vol (%)', 'Max DD (%)', 'Total Return (%)'];
  const sVals = [fm.strategy.cagr, fm.strategy.sharpe, fm.strategy.ann_vol, fm.strategy.max_dd, fm.strategy.total_return];
  const bVals = [fm.benchmark.cagr, fm.benchmark.sharpe, fm.benchmark.ann_vol, fm.benchmark.max_dd, fm.benchmark.total_return];

  let html = '';
  const mLabels = ['CAGR','Sharpe Ratio','Ann. Volatility','Max Drawdown','Total Return'];
  const mClasses = ['blue','purple','green','red','blue'];
  mLabels.forEach((l, i) => {{
    html += `<div class="metric-card ${{mClasses[i]}}"><div class="label">${{l}}</div><div class="value">${{sVals[i]}}${{i!==1?'%':''}}</div><div class="sub">Benchmark: ${{bVals[i]}}${{i!==1?'%':''}}</div></div>`;
  }});
  document.getElementById('bt-metrics').innerHTML = html;

  const bt = D.backtest_results;
  let cumS = [], cumB = [], cs = 1, cb = 1;
  bt.forEach(r => {{ cs *= (1 + r.strategy_return/100); cb *= (1 + r.benchmark_return/100); cumS.push(cs); cumB.push(cb); }});
  new Chart(document.getElementById('chart-bt-full'), {{
    type: 'line',
    data: {{
      labels: bt.map(r => r.date),
      datasets: [
        {{ label: 'Strategy', data: cumS, borderColor: '#3b82f6', backgroundColor: 'rgba(59,130,246,0.08)', fill: true, borderWidth: 2, pointRadius: 0, tension: 0.3 }},
        {{ label: 'Benchmark', data: cumB, borderColor: '#8b5cf6', backgroundColor: 'rgba(139,92,246,0.08)', fill: true, borderWidth: 2, pointRadius: 0, tension: 0.3 }},
      ]
    }},
    options: defaultOpts
  }});

  const btCols = ['date','regime_lagged','chosen_sector','strategy_return','benchmark_return','cumulative_strategy','cumulative_benchmark'];
  setupSearch('search-bt', 'count-bt', bt, btCols, 'table-bt', {{
    colorCols: ['strategy_return','benchmark_return'], regimeCol: 'regime_lagged',
    filterSelect: 'filter-bt-regime', filterKey: 'regime_lagged',
  }});
}})();

// ─── OOS BACKTEST TAB ───
(function() {{
  const om = D.oos_metrics;
  const cards = [
    {{ label: 'CAGR', value: om.strategy.cagr + '%', sub: 'Benchmark: ' + om.benchmark.cagr + '%', cls: 'green' }},
    {{ label: 'Sharpe Ratio', value: om.strategy.sharpe, sub: 'Benchmark: ' + om.benchmark.sharpe, cls: 'purple' }},
    {{ label: 'Ann. Volatility', value: om.strategy.ann_vol + '%', sub: 'Benchmark: ' + om.benchmark.ann_vol + '%', cls: 'blue' }},
    {{ label: 'Max Drawdown', value: om.strategy.max_dd + '%', sub: 'Benchmark: ' + om.benchmark.max_dd + '%', cls: 'red' }},
    {{ label: 'Total Return', value: om.strategy.total_return + '%', sub: 'Benchmark: ' + om.benchmark.total_return + '%', cls: 'blue' }},
  ];
  let html = '';
  cards.forEach(c => {{ html += `<div class="metric-card ${{c.cls}}"><div class="label">${{c.label}}</div><div class="value">${{c.value}}</div><div class="sub">${{c.sub}}</div></div>`; }});
  document.getElementById('oos-metrics-cards').innerHTML = html;

  const bt = D.backtest_results.filter(r => r.date >= '2016-01-01');
  let cumS = [], cumB = [], cs = 1, cb = 1;
  bt.forEach(r => {{ cs *= (1 + r.strategy_return/100); cb *= (1 + r.benchmark_return/100); cumS.push(cs); cumB.push(cb); }});
  new Chart(document.getElementById('chart-oos-full'), {{
    type: 'line',
    data: {{
      labels: bt.map(r => r.date),
      datasets: [
        {{ label: 'Strategy (OOS)', data: cumS, borderColor: '#10b981', backgroundColor: 'rgba(16,185,129,0.08)', fill: true, borderWidth: 2, pointRadius: 0, tension: 0.3 }},
        {{ label: 'Benchmark', data: cumB, borderColor: '#06b6d4', backgroundColor: 'rgba(6,182,212,0.08)', fill: true, borderWidth: 2, pointRadius: 0, tension: 0.3 }},
      ]
    }},
    options: defaultOpts
  }});
}})();
</script>
</body>
</html>"""

# Write the HTML dashboard
dashboard_path = ROOT / "dashboard.html"
with open(dashboard_path, "w", encoding="utf-8") as f:
    f.write(HTML)

print(f"Dashboard saved to: {dashboard_path}")

# ---------------------------------------------------------------------------
# 4.  SERVE LOCALLY & OPEN BROWSER
# ---------------------------------------------------------------------------
PORT = 8050

class QuietHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)
    def log_message(self, format, *args):
        pass  # suppress logs

def open_browser():
    webbrowser.open(f"http://127.0.0.1:{PORT}/dashboard.html")

print(f"\n🚀  Dashboard is live at:  http://127.0.0.1:{PORT}/dashboard.html")
print(f"    Press Ctrl+C to stop the server.\n")

threading.Timer(1.5, open_browser).start()

httpd = HTTPServer(("127.0.0.1", PORT), QuietHandler)
try:
    httpd.serve_forever()
except KeyboardInterrupt:
    print("\n🛑  Server stopped.")
    httpd.server_close()
