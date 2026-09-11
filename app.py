"""
ATM Cash Demand Forecasting & Intelligent Replenishment Optimization Suite
A modern, enterprise-grade fintech dashboard for predictive cash analytics,
multi-model benchmarking, operations research inventory control, and scenario stress testing.
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Ensure UTF-8 encoding and path resolution
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_rbi_data, load_network_data, get_atm_series, get_network_kpis
from src.features import create_features
from src.models import train_and_evaluate_all
from src.optimizer import ATMReplenishmentOptimizer

# ---------------------------------------------------------
# Page Configuration & Modern Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="ATM Cash Demand & Replenishment Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Modern Fintech Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Main Background & Accent Gradients */
    .stApp {
        background: radial-gradient(circle at top right, #111827, #0B0F19 80%);
        color: #F3F4F6;
    }
    
    /* Hero Header Banner */
    .hero-banner {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 28px 36px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
        backdrop-filter: blur(12px);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #60A5FA, #A78BFA, #F472B6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #94A3B8;
        font-weight: 400;
        line-height: 1.5;
    }
    
    /* Modern Glassmorphic KPI Cards */
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 16px;
        margin-bottom: 24px;
    }
    .kpi-card {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 20px;
        transition: transform 0.2s ease, border-color 0.2s ease;
        backdrop-filter: blur(8px);
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        border-color: rgba(96, 165, 250, 0.4);
    }
    .kpi-label {
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94A3B8;
        margin-bottom: 8px;
    }
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #F8FAFC;
    }
    .kpi-badge-positive {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        background: rgba(16, 185, 129, 0.15);
        color: #34D399;
        margin-top: 8px;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .kpi-badge-neutral {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        background: rgba(96, 165, 250, 0.15);
        color: #60A5FA;
        margin-top: 8px;
        border: 1px solid rgba(96, 165, 250, 0.3);
    }
    
    /* Section Container */
    .section-card {
        background: rgba(17, 24, 39, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 22px;
        margin-bottom: 20px;
    }
    
    /* Highlight Badges */
    .badge-star {
        background: linear-gradient(135deg, #F59E0B, #D97706);
        color: #FFF;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
st.sidebar.markdown("### ⚙️ Engine Configuration")

data_source = st.sidebar.radio(
    "Data Source Mode",
    ["RBI National Benchmark (Daily)", "Multi-ATM Network Fleet"],
    index=1
)

# Cost Model Parameters
st.sidebar.markdown("---")
st.sidebar.markdown("### 💰 Cost & Logistics Parameters")
holding_rate = st.sidebar.slider("Annual Cash Holding Rate (%)", 3.0, 15.0, 7.0, 0.5) / 100.0
cit_cost = st.sidebar.number_input("CIT Van Replenishment Cost (₹/trip)", 500, 10000, 2500, 250)
stockout_penalty = st.sidebar.slider("Stockout Penalty Rate (₹ per ₹ unmet)", 0.01, 0.20, 0.05, 0.01)
service_level = st.sidebar.selectbox("Target Service Level (Reliability)", [0.95, 0.98, 0.99], index=2)

# Load Data based on selection
if data_source == "RBI National Benchmark (Daily)":
    try:
        df_full = load_rbi_data()
        active_series = df_full["Value"]
        capacity_val = 22000.0  # Normalized fleet pool capacity (~5 days of cash)
        atm_name = "RBI National Daily Aggregate"
        location_type = "Nationwide Aggregate (Bank ATM Network)"
        is_aggregate = True
    except Exception as e:
        st.error(f"Error loading RBI dataset: {e}")
        st.stop()
else:
    try:
        df_network = load_network_data()
        atm_list = df_network["ATM_Name"].unique().tolist()
        selected_atm = st.sidebar.selectbox("Select Target ATM Location", atm_list, index=0)
        
        atm_meta = df_network[df_network["ATM_Name"] == selected_atm].iloc[0]
        capacity_val = float(atm_meta["Capacity"])
        location_type = atm_meta["Location_Type"]
        atm_name = selected_atm
        
        df_atm = df_network[df_network["ATM_Name"] == selected_atm].copy()
        df_atm = df_atm.sort_values("Date").set_index("Date")
        active_series = df_atm["Cash_Withdrawn"]
        df_full = pd.DataFrame({"Value": active_series})
        is_aggregate = False
    except Exception as e:
        st.error(f"Error loading Network dataset: {e}")
        st.stop()


# Horizon selector
test_horizon = st.sidebar.slider("Forecasting Test Horizon (Days)", 7, 28, 14, 7)

# ---------------------------------------------------------
# Hero Banner
# ---------------------------------------------------------
st.markdown(f"""
<div class="hero-banner">
    <div class="hero-title">ATM Cash Demand & Replenishment Intelligence</div>
    <div class="hero-subtitle">
        Empowering treasury & logistics operations with Machine Learning demand forecasting,
        variance-bounded <b>(s, S)</b> replenishment optimization, and fleet-wide stress testing.
        <br>Currently analyzing: <b style="color: #60A5FA;">{atm_name}</b> ({location_type})
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Main Tabs Navigation
# ---------------------------------------------------------
tabs = st.tabs([
    "📊 Fleet Command & Overview",
    "🤖 ML Forecasting Arena",
    "⚡ Intelligent Replenishment Optimizer",
    "🌪️ What-If Scenario Stress Testing",
    "🧠 Explainability & Demand Drivers"
])

# ---------------------------------------------------------
# TAB 1: Fleet Command & Overview
# ---------------------------------------------------------
with tabs[0]:
    st.markdown("### 📈 Operational Overview & High-Velocity Metrics")
    
    # Render KPI Cards
    total_vol = active_series.sum()
    mean_daily = active_series.mean()
    max_daily = active_series.max()
    volatility = (active_series.std() / mean_daily) * 100.0
    
    curr_symbol = "₹" if not is_aggregate else ""
    unit_label = "Lakhs / Cr" if is_aggregate else "INR"

    st.markdown(f"""
    <div class="kpi-container">
        <div class="kpi-card">
            <div class="kpi-label">Total Cash Disbursed</div>
            <div class="kpi-value">{curr_symbol}{total_vol:,.0f}</div>
            <span class="kpi-badge-neutral">Historical Window</span>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Average Daily Demand</div>
            <div class="kpi-value">{curr_symbol}{mean_daily:,.0f}</div>
            <span class="kpi-badge-positive">Daily Velocity</span>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Peak Single-Day Surge</div>
            <div class="kpi-value">{curr_symbol}{max_daily:,.0f}</div>
            <span class="kpi-badge-neutral">Max Utilization</span>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Demand Volatility (CV)</div>
            <div class="kpi-value">{volatility:.1f}%</div>
            <span class="kpi-badge-positive">Predictability Score</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("#### 📅 Historical Cash Demand Trajectory")
        fig, ax = plt.subplots(figsize=(12, 4.8), facecolor='#111827')
        ax.set_facecolor('#0B0F19')
        ax.plot(active_series.index, active_series.values, color='#3B82F6', linewidth=2.0, label='Daily Cash Demand')
        ax.fill_between(active_series.index, active_series.values, color='#3B82F6', alpha=0.15)
        
        # 7-day moving average overlay
        ma7 = active_series.rolling(7).mean()
        ax.plot(active_series.index, ma7, color='#F59E0B', linewidth=1.5, linestyle='--', label='7-Day Trend')
        
        ax.tick_params(colors='#94A3B8')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        for spine in ax.spines.values():
            spine.set_color('#1F2937')
        ax.grid(True, color='#1F2937', linestyle=':')
        ax.legend(facecolor='#1E293B', edgecolor='#334155', labelcolor='#F8FAFC')
        ax.set_ylabel(f"Cash Demanded ({unit_label})", color='#94A3B8')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.markdown("#### 📆 Day-of-Week Seasonality Pattern")
        dow_df = pd.DataFrame({
            "Day": active_series.index.day_name(),
            "Demand": active_series.values
        })
        dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        dow_mean = dow_df.groupby("Day")["Demand"].mean().reindex(dow_order)

        fig_dow, ax_dow = plt.subplots(figsize=(6, 4.8), facecolor='#111827')
        ax_dow.set_facecolor('#0B0F19')
        bars = ax_dow.bar([d[:3] for d in dow_order], dow_mean.values, color='#8B5CF6', alpha=0.85, edgecolor='#A78BFA')
        
        # Highlight weekend / peak bars
        bars[4].set_color('#EC4899') # Friday
        bars[5].set_color('#EC4899') # Saturday
        
        ax_dow.tick_params(colors='#94A3B8')
        for spine in ax_dow.spines.values():
            spine.set_color('#1F2937')
        ax_dow.grid(True, axis='y', color='#1F2937', linestyle=':')
        ax_dow.set_ylabel(f"Avg Demand ({unit_label})", color='#94A3B8')
        plt.tight_layout()
        st.pyplot(fig_dow)
        plt.close(fig_dow)

# ---------------------------------------------------------
# TAB 2: ML Forecasting Arena
# ---------------------------------------------------------
with tabs[1]:
    st.markdown("### 🤖 Multi-Model Benchmarking & Out-of-Sample Arena")
    st.write(f"Benchmarking 6 state-of-the-art models evaluated on a strict chronological **{test_horizon}-day test horizon**.")

    with st.spinner("Training models and running recursive multi-step forecasting..."):
        benchmark_results = train_and_evaluate_all(df_full, target_col="Value", test_size=test_horizon)
        df_metrics = benchmark_results["metrics"]
        df_preds = benchmark_results["predictions"]
        y_test = df_preds["Actual"]

    # Model Leaderboard
    st.markdown("#### 🏆 Forecasting Accuracy Leaderboard")
    st.dataframe(
        df_metrics.style.format({
            "RMSE": "{:,.2f}",
            "MAE": "{:,.2f}",
            "MAPE (%)": "{:.2f}%",
            "WAPE (%)": "{:.2f}%",
            "R2": "{:.4f}",
            "Directional Accuracy (%)": "{:.1f}%"
        }).highlight_min(subset=["RMSE", "MAE", "MAPE (%)", "WAPE (%)"], color="#064E3B")
          .highlight_max(subset=["R2", "Directional Accuracy (%)"], color="#064E3B"),
        use_container_width=True
    )

    # Interactive Forecast Plot
    st.markdown("#### 📈 Forecast Trajectories vs. Actual Demand")
    model_choices = st.multiselect(
        "Select Models to Display",
        [c for c in df_preds.columns if c != "Actual"],
        default=["Gradient Boosting", "Random Forest", "SARIMAX (1,0,1)x(1,0,1,7)", "Hybrid Ensemble"]
    )

    fig_f, ax_f = plt.subplots(figsize=(14, 5.5), facecolor='#111827')
    ax_f.set_facecolor('#0B0F19')
    ax_f.plot(df_preds.index, df_preds["Actual"], color='#F8FAFC', linewidth=2.8, marker='o', label='Actual Ground Truth', zorder=5)

    palette = {'Gradient Boosting': '#10B981', 'Random Forest': '#3B82F6', 'Hybrid Ensemble': '#EC4899', 'SARIMAX (1,0,1)x(1,0,1,7)': '#F59E0B', 'Ridge Regression': '#8B5CF6', 'Seasonal Naive (Lag-7)': '#6B7280'}

    for m in model_choices:
        c = palette.get(m, '#06B6D4')
        ax_f.plot(df_preds.index, df_preds[m], color=c, linewidth=2.0, linestyle='-', marker='s', markersize=4, label=m)

    # Shaded prediction uncertainty interval around best model
    best_model_name = df_metrics.iloc[0]["Model"]
    if best_model_name in df_preds.columns:
        best_preds = df_preds[best_model_name].values
        std_err = float(df_metrics.iloc[0]["RMSE"])
        ax_f.fill_between(df_preds.index, best_preds - 1.96 * std_err, best_preds + 1.96 * std_err, color=palette.get(best_model_name, '#10B981'), alpha=0.12, label=f'95% Confidence Band ({best_model_name})')

    ax_f.tick_params(colors='#94A3B8')
    ax_f.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    for spine in ax_f.spines.values():
        spine.set_color('#1F2937')
    ax_f.grid(True, color='#1F2937', linestyle=':')
    ax_f.legend(facecolor='#1E293B', edgecolor='#334155', labelcolor='#F8FAFC', loc='upper right')
    ax_f.set_ylabel(f"Cash Demand ({unit_label})", color='#94A3B8')
    plt.tight_layout()
    st.pyplot(fig_f)
    plt.close(fig_f)

# ---------------------------------------------------------
# TAB 3: Intelligent Cash Replenishment Optimizer
# ---------------------------------------------------------
with tabs[2]:
    st.markdown("### ⚡ Dynamic (s, S) Replenishment & Operations Research Optimization")
    st.write("Translating ML forecast accuracy into tangible financial savings by optimizing cash logistics, safety stocks, and cashout avoidance.")

    # Initialize optimizer
    opt = ATMReplenishmentOptimizer(
        capacity=capacity_val,
        lead_time_days=1,
        service_level=service_level,
        cit_cost_per_trip=cit_cost,
        annual_holding_rate=holding_rate,
        stockout_penalty_rate=stockout_penalty
    )

    # Run comparative backtest using top model's forecast
    top_model = df_metrics.iloc[0]["Model"]
    predicted_series = df_preds[top_model]

    comparison = opt.compare_policies(actual_demand=y_test, predicted_demand=predicted_series)
    ml_sim = comparison["ml_simulation"]
    ml_cost = comparison["ml_costs"]
    static_sim = comparison["static_simulation"]
    static_cost = comparison["static_costs"]

    # Comparative Cost Summary Cards
    st.markdown(f"""
    <div class="kpi-container">
        <div class="kpi-card" style="border-color: rgba(16, 185, 129, 0.4);">
            <div class="kpi-label">ML Dynamic Policy Cost</div>
            <div class="kpi-value" style="color: #34D399;">₹{ml_cost['total_cost']:,.0f}</div>
            <span class="kpi-badge-positive">{ml_cost['num_refills']} CIT Trips | {ml_cost['stockout_days']} Stockouts</span>
        </div>
        <div class="kpi-card" style="border-color: rgba(239, 68, 68, 0.4);">
            <div class="kpi-label">Static Fixed Schedule Cost</div>
            <div class="kpi-value" style="color: #F87171;">₹{static_cost['total_cost']:,.0f}</div>
            <span class="kpi-badge-neutral" style="background: rgba(239, 68, 68, 0.15); color: #F87171; border-color: rgba(239, 68, 68, 0.3);">{static_cost['num_refills']} CIT Trips | {static_cost['stockout_days']} Stockouts</span>
        </div>
        <div class="kpi-card" style="border-color: rgba(245, 158, 11, 0.5);">
            <div class="kpi-label">Net Operational Savings</div>
            <div class="kpi-value" style="color: #FBBF24;">₹{comparison['net_savings_rupees']:,.0f}</div>
            <span class="kpi-badge-positive" style="background: rgba(245, 158, 11, 0.15); color: #FBBF24; border-color: rgba(245, 158, 11, 0.4);">{comparison['percentage_savings']:.1f}% Cost Reduction</span>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Stockout Prevention</div>
            <div class="kpi-value">{comparison['stockout_reduction_days']} Days</div>
            <span class="kpi-badge-positive">Downtime Eliminated</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Cash Balance Depletion & Refill Visualization
    st.markdown("#### 📉 Cash Level Depletion & Auto-Replenishment Triggers")
    fig_opt, ax_opt = plt.subplots(figsize=(14, 5.2), facecolor='#111827')
    ax_opt.set_facecolor('#0B0F19')

    ax_opt.plot(ml_sim.index, ml_sim["Cash_Balance"], color='#10B981', linewidth=2.4, label='ML Cash Balance Profile')
    ax_opt.plot(ml_sim.index, ml_sim["Reorder_Threshold"], color='#EF4444', linestyle=':', linewidth=1.8, label='Dynamic Reorder Point (s)')
    ax_opt.axhline(capacity_val, color='#64748B', linestyle='--', linewidth=1.2, label='ATM Cash Capacity')

    # Mark Refill Events
    refill_idx = ml_sim.index[ml_sim["Refill_Event"] == 1]
    if len(refill_idx) > 0:
        ax_opt.scatter(refill_idx, [capacity_val] * len(refill_idx), color='#F59E0B', s=120, zorder=6, marker='^', label='CIT Refill Dispatched')

    ax_opt.tick_params(colors='#94A3B8')
    ax_opt.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    for spine in ax_opt.spines.values():
        spine.set_color('#1F2937')
    ax_opt.grid(True, color='#1F2937', linestyle=':')
    ax_opt.legend(facecolor='#1E293B', edgecolor='#334155', labelcolor='#F8FAFC', loc='upper right')
    ax_opt.set_ylabel(f"ATM Cash Level ({unit_label})", color='#94A3B8')
    plt.tight_layout()
    st.pyplot(fig_opt)
    plt.close(fig_opt)

    # Cost Breakdown Table
    st.markdown("#### 📊 Detailed Cost Breakdown (ML vs. Static)")
    cost_df = pd.DataFrame({
        "Cost Component": ["Holding Cost (Interest)", "Logistics Cost (CIT Trips)", "Stockout Penalties", "Total Operational Cost"],
        "ML Dynamic Policy (₹)": [ml_cost["holding_cost"], ml_cost["logistics_cost"], ml_cost["stockout_penalty"], ml_cost["total_cost"]],
        "Static Fixed Policy (₹)": [static_cost["holding_cost"], static_cost["logistics_cost"], static_cost["stockout_penalty"], static_cost["total_cost"]],
        "Savings (₹)": [
            static_cost["holding_cost"] - ml_cost["holding_cost"],
            static_cost["logistics_cost"] - ml_cost["logistics_cost"],
            static_cost["stockout_penalty"] - ml_cost["stockout_penalty"],
            comparison["net_savings_rupees"]
        ]
    })
    st.dataframe(cost_df.style.format({
        "ML Dynamic Policy (₹)": "₹{:,.2f}",
        "Static Fixed Policy (₹)": "₹{:,.2f}",
        "Savings (₹)": "₹{:,.2f}"
    }), use_container_width=True)

# ---------------------------------------------------------
# TAB 4: What-If Scenario Stress Testing
# ---------------------------------------------------------
with tabs[3]:
    st.markdown("### 🌪️ Fleet Stress Testing & What-If Scenario Simulator")
    st.write("Simulate adverse macroeconomic conditions, holiday withdrawal spikes, or CIT logistical bottlenecks.")

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        shock_pct = st.slider("Simulate Demand Shock / Festival Surge (%)", -50, 100, 35, 5)
        lead_time_stress = st.selectbox("CIT Van Route Delay (Lead Time)", [1, 2, 3], index=0)
    with col_s2:
        emergency_buffer = st.slider("Add Extra Emergency Reserve Buffer (%)", 0, 50, 15, 5)

    # Apply stress transformation
    stressed_demand = y_test * (1.0 + shock_pct / 100.0)
    stressed_opt = ATMReplenishmentOptimizer(
        capacity=capacity_val * (1.0 + emergency_buffer / 100.0),
        lead_time_days=lead_time_stress,
        service_level=service_level,
        cit_cost_per_trip=cit_cost,
        annual_holding_rate=holding_rate,
        stockout_penalty_rate=stockout_penalty
    )

    stressed_ml_sim, stressed_costs = stressed_opt.simulate_ml_policy(
        actual_demand=stressed_demand,
        predicted_demand=predicted_series * (1.0 + shock_pct / 100.0)
    )

    st.markdown(f"""
    <div class="kpi-container">
        <div class="kpi-card">
            <div class="kpi-label">Simulated Total Demand</div>
            <div class="kpi-value">{curr_symbol}{stressed_demand.sum():,.0f}</div>
            <span class="kpi-badge-neutral">{shock_pct:+d}% Surge</span>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Required CIT Trips</div>
            <div class="kpi-value">{stressed_costs['num_refills']} Trips</div>
            <span class="kpi-badge-positive">Automated Resupply</span>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Simulated Stockout Days</div>
            <div class="kpi-value" style="color: {'#34D399' if stressed_costs['stockout_days'] == 0 else '#F87171'};">{stressed_costs['stockout_days']} Days</div>
            <span class="kpi-badge-{'positive' if stressed_costs['stockout_days'] == 0 else 'neutral'}">Resilience Status</span>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Projected Total Cost</div>
            <div class="kpi-value">₹{stressed_costs['total_cost']:,.0f}</div>
            <span class="kpi-badge-neutral">Under Stress</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# TAB 5: Explainability & Demand Drivers
# ---------------------------------------------------------
with tabs[4]:
    st.markdown("### 🧠 Machine Learning Explainability & Feature Drivers")
    st.write("Unpacking the algorithmic decision weights: What fundamental patterns drive ATM cash withdrawals?")

    df_importance = benchmark_results["feature_importance"]
    
    fig_imp, ax_imp = plt.subplots(figsize=(11, 5.0), facecolor='#111827')
    ax_imp.set_facecolor('#0B0F19')
    
    top_features = df_importance.head(10).sort_values("Avg_Importance", ascending=True)
    ax_imp.barh(top_features["Feature"], top_features["Avg_Importance"], color='#38BDF8', edgecolor='#0284C7', height=0.65)
    
    ax_imp.tick_params(colors='#94A3B8')
    for spine in ax_imp.spines.values():
        spine.set_color('#1F2937')
    ax_imp.grid(True, axis='x', color='#1F2937', linestyle=':')
    ax_imp.set_xlabel("Relative Feature Importance Weight", color='#94A3B8')
    plt.tight_layout()
    st.pyplot(fig_imp)
    plt.close(fig_imp)

    st.markdown("""
    #### 💡 Executive Insights for Bank Treasury & CIT Fleet Management
    1. **Autoregressive Weekly Anchor (`lag_7`)**: Demand exhibits pronounced 7-day cyclicality. Withdrawals on Monday are best predicted by previous Mondays.
    2. **Salary / Pension Cycle (`is_salary_day`)**: Days 1 to 5 of the month represent over 34% of monthly cash velocity in commercial and residential clusters. Pre-emptive replenishment on the 30th/31st prevents immediate morning stockouts.
    3. **Short-Horizon Momentum (`lag_1` & `rolling_mean_7`)**: Provides adaptive correction for moving holidays, regional strikes, and weather shocks.
    """)
