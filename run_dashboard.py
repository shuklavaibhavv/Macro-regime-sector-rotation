import os
import sys
import math
import subprocess
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Macro Regime Classification & Sector Rotation Strategy",
    page_icon="📊",
    layout="wide"
)

# Custom Institutional CSS styling (Minimalist Dark Finance Aesthetic)
st.markdown("""
<style>
    /* Global Page Styling */
    .stApp {
        background-color: #0E1117;
        color: #E0E0E0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Headers & Typography */
    h1, h2, h3, h4 {
        color: #F0F2F6 !important;
        font-weight: 600 !important;
        letter-spacing: -0.02em !important;
    }
    
    .main-title {
        font-size: 1.85rem;
        font-weight: 700;
        color: #FFFFFF;
        margin-bottom: 0.2rem;
    }
    
    .sub-title {
        font-size: 0.95rem;
        color: #909090;
        margin-bottom: 1.5rem;
        border-bottom: 1px solid #262930;
        padding-bottom: 0.75rem;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #161922;
        border-right: 1px solid #262930;
    }
    
    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #1A1D27;
        border: 1px solid #2A2E3D;
        padding: 12px 16px;
        border-radius: 6px;
    }
    
    div[data-testid="stMetricLabel"] {
        font-size: 0.8rem !important;
        color: #888888 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 1.35rem !important;
        font-weight: 600 !important;
        color: #E6E6E6 !important;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #262930;
    }

    .stTabs [data-baseweb="tab"] {
        height: 40px;
        white-space: pre;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        color: #888888;
        font-size: 0.9rem;
        font-weight: 500;
        padding: 0px 16px;
    }

    .stTabs [aria-selected="true"] {
        background-color: #1E222D !important;
        color: #3182CE !important;
        border-bottom: 2px solid #3182CE !important;
    }
    
    /* Table Styling */
    table {
        color: #D0D0D0 !important;
    }
</style>
""", unsafe_allow_html=True)

# Main Title Header
st.markdown('<div class="main-title">Macro Regime Classification & Sector Rotation Framework</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Quantitative Research Platform | Federal Reserve Macro Indicators & S&P Sector Rotation (2005–Present)</div>', unsafe_allow_html=True)

# Sidebar Orchestrator & Configuration Panel
st.sidebar.markdown("### Pipeline Orchestrator")
st.sidebar.caption("Execute underlying modular ETL and backtesting scripts:")

if st.sidebar.button("Execute Full Pipeline"):
    with st.spinner("Executing pipeline scripts in order..."):
        try:
            py_bin = sys.executable
            subprocess.run([py_bin, "src/fetch_data.py"], check=True)
            subprocess.run([py_bin, "src/build_regime_history.py"], check=True)
            subprocess.run([py_bin, "src/fetch_sector_data.py"], check=True)
            subprocess.run([py_bin, "src/regime_sector_analysis.py"], check=True)
            subprocess.run([py_bin, "src/backtest_rotation_oos.py"], check=True)
            subprocess.run([py_bin, "src/backtest_rotation_voltarget.py"], check=True)
            st.sidebar.success("Pipeline executed successfully!")
        except Exception as e:
            st.sidebar.error(f"Pipeline execution error: {e}")

st.sidebar.markdown("---")
st.sidebar.markdown("### Strategy Model Parameters")
st.sidebar.markdown("""
**In-Sample Rule Set (2005–2015):**
- **Expansion:** `XLE` (Energy)
- **Slowdown:** `XLB` (Materials)
- **Recovery:** `XLP` (Staples)
- **Contraction:** `XLK` (Technology)

**Execution Engine:**
- **Execution Lag:** 1 Month (`t-1` signal)
- **OOS Test Split:** 2016–2026 (10.5 Yrs)
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### Research Documentation")
if os.path.exists("JPMC_Quantitative_Interview_100_QA.pdf"):
    with open("JPMC_Quantitative_Interview_100_QA.pdf", "rb") as pdf_file:
        st.sidebar.download_button(
            label="Download 100-QA Interview Guide (PDF)",
            data=pdf_file,
            file_name="JPMC_Quantitative_Interview_100_QA.pdf",
            mime="application/pdf"
        )

# Load Datasets
@st.cache_data
def load_data():
    regime_df = pd.read_csv("data/regime_history.csv")
    sector_df = pd.read_csv("data/sector_returns.csv")
    merged_df = pd.read_csv("data/regime_sector_merged.csv")
    summary_df = pd.read_csv("data/regime_sector_summary.csv", index_col=0)
    backtest_df = pd.read_csv("data/backtest_results.csv")
    return regime_df, sector_df, merged_df, summary_df, backtest_df

try:
    regime_df, sector_df, merged_df, summary_df, backtest_df = load_data()
except Exception as e:
    st.error("Data files not found. Please click 'Execute Full Pipeline' in the sidebar.")
    st.stop()

# Define Color Palette for Regimes
REGIME_COLORS = {
    'Expansion': '#22C55E',   # Emerald Green
    'Slowdown': '#F59E0B',    # Amber
    'Recovery': '#3B82F6',    # Blue
    'Contraction': '#EF4444'   # Crimson Red
}
SECTORS = ['XLF', 'XLK', 'XLE', 'XLV', 'XLY', 'XLP', 'XLI', 'XLU', 'XLB']

# Create Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Macro Diagnostics", 
    "Sector Factor Attribution", 
    "Intra-Recession Dynamics", 
    "Out-of-Sample Backtest", 
    "Allocation Simulator"
])

# ==========================================
# TAB 1: MACRO DIAGNOSTICS & TRANSITION MATRIX
# ==========================================
with tab1:
    st.markdown("#### Macroeconomic Indicators & Regime Diagnostics")
    
    latest = merged_df.iloc[-1]
    prev = merged_df.iloc[-2]
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Observation Date", str(latest['date']))
    col2.metric("Macro Regime", str(latest['regime']))
    col3.metric("PMI Proxy Index", f"{latest['pmi']:.2f}", delta=f"{latest['pmi'] - prev['pmi']:.2f} pts")
    col4.metric("Yield Spread (10Y-2Y)", f"{latest['yield_curve_spread']:.2f}%", delta=f"{latest['yield_curve_spread'] - prev['yield_curve_spread']:.2f}%")
    
    st.markdown("---")
    
    # Dual Axis Plotly Chart
    fig_macro = go.Figure()
    fig_macro.add_trace(go.Scatter(
        x=merged_df['date'], y=merged_df['pmi'],
        mode='lines', name='PMI Proxy Index',
        line=dict(color='#3B82F6', width=2)
    ))
    fig_macro.add_trace(go.Scatter(
        x=merged_df['date'], y=merged_df['yield_curve_spread'],
        mode='lines', name='10Y-2Y Yield Spread (%)', yaxis='y2',
        line=dict(color='#F59E0B', width=1.5, dash='dot')
    ))
    
    fig_macro.add_shape(
        type='line', x0=merged_df['date'].iloc[0], y0=50, x1=merged_df['date'].iloc[-1], y1=50,
        line=dict(color='#666666', dash='dash', width=1)
    )
    
    fig_macro.update_layout(
        title="Historical Macro Series: Manufacturing Output & Term Structure",
        xaxis=dict(title="", showgrid=False),
        yaxis=dict(title="PMI Proxy Index (Neutral = 50)", showgrid=True, gridcolor='#222630'),
        yaxis2=dict(title="Yield Curve Spread (%)", overlaying='y', side='right', showgrid=False),
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(0,0,0,0)'),
        paper_bgcolor='#0E1117',
        plot_bgcolor='#12151E',
        font=dict(color='#C0C0C0'),
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig_macro, width='stretch')
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("##### Historical Regime Distribution (2005–Present)")
        counts = merged_df['regime'].value_counts()
        fig_pie = px.pie(
            names=counts.index, values=counts.values,
            color=counts.index, color_discrete_map=REGIME_COLORS,
            hole=0.45
        )
        fig_pie.update_layout(
            paper_bgcolor='#0E1117', plot_bgcolor='#0E1117',
            font=dict(color='#C0C0C0'), height=320,
            margin=dict(l=10, r=10, t=20, b=20)
        )
        st.plotly_chart(fig_pie, width='stretch')
        
    with col_right:
        st.markdown("##### Markov Regime Transition Probability Matrix")
        # Compute 1-step regime transition matrix P(t+1 | t)
        regimes_order = ['Expansion', 'Slowdown', 'Contraction', 'Recovery']
        merged_df['next_regime'] = merged_df['regime'].shift(-1)
        transition_counts = pd.crosstab(merged_df['regime'], merged_df['next_regime'], normalize='index') * 100
        transition_counts = transition_counts.reindex(index=regimes_order, columns=regimes_order).fillna(0).round(1)
        
        fig_trans = px.imshow(
            transition_counts, text_auto=True,
            color_continuous_scale="Blues",
            labels=dict(x="Regime (t+1)", y="Regime (t)", color="Probability (%)")
        )
        fig_trans.update_layout(
            paper_bgcolor='#0E1117', plot_bgcolor='#0E1117',
            font=dict(color='#C0C0C0'), height=320,
            margin=dict(l=10, r=10, t=20, b=20)
        )
        st.plotly_chart(fig_trans, width='stretch')

# ==========================================
# TAB 2: SECTOR FACTOR ATTRIBUTION
# ==========================================
with tab2:
    st.markdown("#### Sector Return & Risk Attribution across Regimes")
    
    avg_returns = merged_df.groupby('regime')[SECTORS].mean().round(2)
    win_rates = merged_df.groupby('regime')[SECTORS].apply(lambda g: (g > 0).mean() * 100).round(1)
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("##### Average Monthly Return (%) by Regime")
        fig_heat1 = px.imshow(
            avg_returns, text_auto=True,
            color_continuous_scale="RdYlGn",
            labels=dict(x="Sector ETF", y="Macro Regime", color="Return (%)")
        )
        fig_heat1.update_layout(
            paper_bgcolor='#0E1117', plot_bgcolor='#0E1117',
            font=dict(color='#C0C0C0'), height=350
        )
        st.plotly_chart(fig_heat1, width='stretch')
        
    with col_b:
        st.markdown("##### Sector Win Rate (%) by Regime")
        fig_heat2 = px.imshow(
            win_rates, text_auto=True,
            color_continuous_scale="Blues",
            labels=dict(x="Sector ETF", y="Macro Regime", color="Win Rate (%)")
        )
        fig_heat2.update_layout(
            paper_bgcolor='#0E1117', plot_bgcolor='#0E1117',
            font=dict(color='#C0C0C0'), height=350
        )
        st.plotly_chart(fig_heat2, width='stretch')

    st.markdown("---")
    st.markdown("##### Risk-Return Profile across Sector Universe (2005–Present)")
    
    # Calculate full-sample annual CAGR & Volatility per sector
    sector_stats = []
    for sec in SECTORS:
        cum_val = (1 + merged_df[sec] / 100).cumprod().iloc[-1]
        cagr = (cum_val ** (12 / len(merged_df)) - 1) * 100
        vol = merged_df[sec].std() * np.sqrt(12)
        sharpe = cagr / vol
        sector_stats.append({'Sector': sec, 'CAGR (%)': cagr, 'Volatility (%)': vol, 'Sharpe': sharpe})
        
    stats_df = pd.DataFrame(sector_stats)
    
    fig_scatter = px.scatter(
        stats_df, x='Volatility (%)', y='CAGR (%)', text='Sector', size='Sharpe',
        color='Sharpe', color_continuous_scale='Viridis',
        title="Annualized Risk vs. Return per Sector ETF"
    )
    fig_scatter.update_traces(textposition='top center', marker=dict(size=14))
    fig_scatter.update_layout(
        paper_bgcolor='#0E1117', plot_bgcolor='#12151E',
        font=dict(color='#C0C0C0'), height=400
    )
    st.plotly_chart(fig_scatter, width='stretch')

# ==========================================
# TAB 3: INTRA-RECESSION DYNAMICS
# ==========================================
with tab3:
    st.markdown("#### Intra-Recession Dynamics: Early vs. Late Contraction Sub-Phases")
    st.markdown("""
    **Institutional Research Question:** Is Technology (`XLK`) outperformance during economic downturns driven by early quality/balance-sheet resilience or late-stage monetary easing rallies?
    """)
    
    m_df = merged_df.copy()
    m_df['run_id'] = (m_df['regime'] != m_df['regime'].shift()).cumsum()
    c_df = m_df[m_df['regime'] == 'Contraction'].copy()
    
    sub_labels = []
    for rid in c_df['run_id'].unique():
        rows = c_df[c_df['run_id'] == rid]
        n = len(rows)
        early_n = math.ceil(n / 2)
        sub_labels.extend(['Early Contraction'] * early_n + ['Late Contraction'] * (n - early_n))
        
    c_df['sub_phase'] = sub_labels
    early_late_returns = c_df.groupby('sub_phase')[SECTORS].mean().round(2)
    
    fig_bar = px.bar(
        early_late_returns.T.reset_index(),
        x='index', y=['Early Contraction', 'Late Contraction'],
        barmode='group',
        labels={'index': 'Sector ETF', 'value': 'Average Monthly Return (%)'},
        title="Sector Performance Shift: Initial Shock vs. Late Recovery Phase"
    )
    fig_bar.update_layout(
        paper_bgcolor='#0E1117', plot_bgcolor='#12151E',
        font=dict(color='#C0C0C0'), height=420,
        legend=dict(title="", x=0.01, y=0.99)
    )
    st.plotly_chart(fig_bar, width='stretch')
    
    st.info("""
    **Key Takeaways:**
    - **Early Contraction (Initial Panic):** Cyclical sectors crash—Financials (`XLF` **-0.60%/mo**), Industrials (`XLI` **-0.17%/mo**), and Utilities (`XLU` **-0.56%/mo**). Tech (`XLK`) generates **+2.01%/mo** due to cash balance sheets.
    - **Late Contraction (Monetary Easing Rally):** Federal Reserve rate cuts trigger broad recovery rallies: Financials (`XLF` **+2.97%/mo**), Tech (`XLK` **+2.84%/mo**), and Discretionary (`XLY` **+2.74%/mo**).
    """)

# ==========================================
# TAB 4: OUT-OF-SAMPLE BACKTEST & RISK ENGINE
# ==========================================
with tab4:
    st.markdown("#### Out-of-Sample Performance & Risk Diagnostics (2016–Present)")
    
    # Filter controls
    backtest_df['date_dt'] = pd.to_datetime(backtest_df['date'])
    min_date = backtest_df['date_dt'].min()
    max_date = backtest_df['date_dt'].max()
    
    date_range = st.slider(
        "Select Backtest Time Window:",
        min_value=min_date.to_pydatetime(),
        max_value=max_date.to_pydatetime(),
        value=(min_date.to_pydatetime(), max_date.to_pydatetime()),
        format="YYYY-MM"
    )
    
    filtered_bt = backtest_df[
        (backtest_df['date_dt'] >= date_range[0]) & 
        (backtest_df['date_dt'] <= date_range[1])
    ].copy()
    
    filtered_bt['cum_strat'] = (1 + filtered_bt['strategy_return'] / 100).cumprod()
    filtered_bt['cum_bench'] = (1 + filtered_bt['benchmark_return'] / 100).cumprod()
    
    fig_equity = go.Figure()
    fig_equity.add_trace(go.Scatter(
        x=filtered_bt['date'], y=filtered_bt['cum_strat'],
        mode='lines', name='OOS Macro Sector Rotation Strategy',
        line=dict(color='#22C55E', width=2.5)
    ))
    fig_equity.add_trace(go.Scatter(
        x=filtered_bt['date'], y=filtered_bt['cum_bench'],
        mode='lines', name='Equal-Weight 9-Sector Benchmark',
        line=dict(color='#888888', width=1.5, dash='dash')
    ))
    
    fig_equity.update_layout(
        title="Compounded Portfolio Growth ($1.00 Base)",
        xaxis=dict(title="", showgrid=False),
        yaxis=dict(title="Growth of $1.00 ($)", showgrid=True, gridcolor='#222630'),
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(0,0,0,0)'),
        paper_bgcolor='#0E1117', plot_bgcolor='#12151E',
        font=dict(color='#C0C0C0'), height=420
    )
    st.plotly_chart(fig_equity, width='stretch')
    
    # Calculate performance metrics
    n_months = len(filtered_bt)
    if n_months > 0:
        s_tot = (filtered_bt['cum_strat'].iloc[-1] - 1) * 100
        b_tot = (filtered_bt['cum_bench'].iloc[-1] - 1) * 100
        
        s_cagr = ((filtered_bt['cum_strat'].iloc[-1]) ** (12 / n_months) - 1) * 100
        b_cagr = ((filtered_bt['cum_bench'].iloc[-1]) ** (12 / n_months) - 1) * 100
        
        s_vol = filtered_bt['strategy_return'].std() * np.sqrt(12)
        b_vol = filtered_bt['benchmark_return'].std() * np.sqrt(12)
        
        s_sharpe = s_cagr / s_vol if s_vol > 0 else 0
        b_sharpe = b_cagr / b_vol if b_vol > 0 else 0
        
        # Max Drawdown
        s_peaks = filtered_bt['cum_strat'].cummax()
        s_dd = ((filtered_bt['cum_strat'] - s_peaks) / s_peaks).min() * 100
        
        b_peaks = filtered_bt['cum_bench'].cummax()
        b_dd = ((filtered_bt['cum_bench'] - b_peaks) / b_peaks).min() * 100
        
        metrics_table = pd.DataFrame({
            'Performance Metric': ['Total Return (%)', 'CAGR (%)', 'Annualized Volatility (%)', 'Sharpe Ratio (Rf=0)', 'Max Drawdown (%)'],
            'Strategy (OOS)': [f"{s_tot:.2f}%", f"{s_cagr:.2f}%", f"{s_vol:.2f}%", f"{s_sharpe:.2f}", f"{s_dd:.2f}%"],
            'Benchmark (Equal-Weight)': [f"{b_tot:.2f}%", f"{b_cagr:.2f}%", f"{b_vol:.2f}%", f"{b_sharpe:.2f}", f"{b_dd:.2f}%"]
        })
        st.table(metrics_table)

# ==========================================
# TAB 5: ALLOCATION SIMULATOR
# ==========================================
with tab5:
    st.markdown("#### Interactive Strategy Allocation Simulator")
    st.markdown("Customize sector mappings per regime to simulate real-time out-of-sample performance:")
    
    sim_col1, sim_col2, sim_col3, sim_col4 = st.columns(4)
    with sim_col1:
        exp_sec = st.selectbox("Expansion Allocation", SECTORS, index=SECTORS.index('XLE'))
    with sim_col2:
        slow_sec = st.selectbox("Slowdown Allocation", SECTORS, index=SECTORS.index('XLB'))
    with sim_col3:
        rec_sec = st.selectbox("Recovery Allocation", SECTORS, index=SECTORS.index('XLP'))
    with sim_col4:
        con_sec = st.selectbox("Contraction Allocation", SECTORS, index=SECTORS.index('XLK'))
        
    # Simulate custom strategy returns on test set (2016-present)
    test_df = merged_df[merged_df['date'] >= '2016-01-01'].copy()
    test_df['lagged_regime'] = test_df['regime'].shift(1)
    test_df = test_df.dropna(subset=['lagged_regime'])
    
    custom_map = {
        'Expansion': exp_sec,
        'Slowdown': slow_sec,
        'Recovery': rec_sec,
        'Contraction': con_sec
    }
    
    sim_returns = []
    for idx, row in test_df.iterrows():
        r = row['lagged_regime']
        target_sec = custom_map.get(r, 'XLK')
        sim_returns.append(row[target_sec])
        
    test_df['sim_return'] = sim_returns
    test_df['sim_cum'] = (1 + test_df['sim_return'] / 100).cumprod()
    
    sim_tot = (test_df['sim_cum'].iloc[-1] - 1) * 100
    sim_cagr = ((test_df['sim_cum'].iloc[-1]) ** (12 / len(test_df)) - 1) * 100
    sim_vol = test_df['sim_return'].std() * np.sqrt(12)
    sim_sharpe = sim_cagr / sim_vol if sim_vol > 0 else 0
    
    st.markdown("---")
    res1, res2, res3, res4 = st.columns(4)
    res1.metric("Simulated Total Return", f"{sim_tot:.2f}%")
    res2.metric("Simulated CAGR", f"{sim_cagr:.2f}%")
    res3.metric("Simulated Volatility", f"{sim_vol:.2f}%")
    res4.metric("Simulated Sharpe Ratio", f"{sim_sharpe:.2f}")
    
    fig_sim = go.Figure()
    fig_sim.add_trace(go.Scatter(
        x=test_df['date'], y=test_df['sim_cum'],
        mode='lines', name='Simulated Strategy Allocation',
        line=dict(color='#3B82F6', width=2.5)
    ))
    fig_sim.update_layout(
        title="Simulated Portfolio Growth ($1.00 Base)",
        paper_bgcolor='#0E1117', plot_bgcolor='#12151E',
        font=dict(color='#C0C0C0'), height=380
    )
    st.plotly_chart(fig_sim, width='stretch')
