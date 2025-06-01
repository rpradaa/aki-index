import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- DARK THEME & CUSTOM STYLES ---
st.markdown(
    """
    <style>
    .stApp { background-color: #0e1117; color: #fafafa; }
    .section-header {
        color: #1a73e8;
        font-weight: 600;
        margin-top: 2em;
        font-size: 22px;
    }
    .advisor-feedback {
        background: #1f2937;
        border-left: 5px solid #1a73e8;
        padding: 1em;
        margin-bottom: 1em;
        border-radius: 8px;
        color: #e0e7ff;
    }
    .alt-suggestion {
        background: #3b4252;
        border-left: 5px solid #fbbf24;
        padding: 1em;
        margin-bottom: 1em;
        border-radius: 8px;
        color: #fffacd;
    }
    .peer-feedback {
        background: #2e3440;
        border-left: 5px solid #88c0d0;
        padding: 1em;
        margin-bottom: 1em;
        border-radius: 8px;
        color: #d8dee9;
    }
    .metric-label { color: #a0aec0; font-weight: 500; }
    .metric-value { color: #fafafa; font-weight: 700; font-size: 28px; }
    .streamlit-expanderHeader {
        color: #63b3ed;
        font-weight: 600;
    }
    a { color: #63b3ed; text-decoration: none; }
    a:hover { text-decoration: underline; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- HEADER ---
st.title("Step 2: Foundation — Agentic AI Portfolio Builder")
st.markdown("""
Build your own tech index, see how it stacks up against the S&P 500, and get **personalized, AI-powered financial advice**—just like a real portfolio manager.
""")

# --- SIDEBAR: NAVIGATION & PROFILE ---
with st.sidebar:
    st.header("Investor Profile")
    risk_profile = st.selectbox(
        "Risk Tolerance",
        ["Conservative", "Moderate", "Aggressive"],
        help="AI will optimize and critique your portfolio based on this."
    )
    investment_goal = st.selectbox(
        "Investment Goal",
        ["Steady Growth", "Aggressive Growth", "Capital Preservation", "Income"],
        help="Helps tailor AI feedback."
    )
    st.markdown("---")
    st.header("Portfolio Construction")

# --- STOCK SELECTION ---
magnificent_7 = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META', 'NVDA', 'TSLA']
selected_stocks = st.sidebar.multiselect(
    "Select 1-7 Tech Stocks",
    options=magnificent_7,
    default=magnificent_7[:3]
)
if not selected_stocks:
    st.warning("Select at least 1 stock.")
    st.stop()

# --- WEIGHTING METHOD ---
weight_method = st.sidebar.radio(
    "Weighting Method",
    ["Equal Weight", "Manual"],
    help="Choose how to assign weights to your selected stocks."
)

# --- WEIGHTS LOGIC ---
if weight_method == "Equal Weight":
    weights = np.repeat(1/len(selected_stocks), len(selected_stocks))
else:
    weights = []
    st.sidebar.markdown("Enter custom weights (must sum to 1):")
    for stock in selected_stocks:
        w = st.sidebar.number_input(
            f"Weight for {stock}",
            min_value=0.0,
            max_value=1.0,
            value=round(1/len(selected_stocks), 2),
            step=0.01,
            key=stock
        )
        weights.append(w)
    weights = np.array(weights)
    if not np.isclose(weights.sum(), 1.0):
        st.warning("Custom weights must sum to 1. Adjust your weights.")
        st.stop()

# --- DATE SELECTION ---
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-12-31"))
if start_date >= end_date:
    st.sidebar.error("Start Date must be before End Date.")
    st.stop()

# --- DATA FETCHING ---
@st.cache_data(show_spinner=True)
def fetch_data(tickers, start, end):
    df = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        if 'Adj Close' in df.columns.get_level_values(0):
            data = df['Adj Close']
        elif 'Close' in df.columns.get_level_values(0):
            data = df['Close']
        else:
            st.error("Neither 'Adj Close' nor 'Close' found in yfinance data.")
            return pd.DataFrame()
    else:
        data = df
    return data

tickers = selected_stocks + ['^GSPC']
data = fetch_data(tickers, start_date, end_date)
if data.empty or data.isnull().all().all():
    st.error("No data found for the selected period. Try a different date range or stocks.")
    st.stop()

norm_prices = data.div(data.iloc[0]) * 100

# --- AI OPTIMIZATION: SUGGESTED WEIGHTS BASED ON RECENT PERFORMANCE ---
def ai_optimize_weights(prices, risk_profile):
    lookback = min(252, len(prices))
    returns = prices.iloc[-1] / prices.iloc[-lookback] - 1
    returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
    if np.allclose(returns, 0):
        weights = np.repeat(1/len(returns), len(returns))
    else:
        if risk_profile == "Conservative":
            raw = np.clip(returns, 0, None)
            if raw.sum() == 0:
                weights = np.repeat(1/len(raw), len(raw))
            else:
                weights = raw / raw.sum()
                weights = np.clip(weights, 0, 0.3)
                weights = weights / weights.sum()
        elif risk_profile == "Moderate":
            raw = np.clip(returns, 0, None)
            if raw.sum() == 0:
                weights = np.repeat(1/len(raw), len(raw))
            else:
                weights = raw / raw.sum()
                weights = np.clip(weights, 0, 0.5)
                weights = weights / weights.sum()
        else:
            raw = np.clip(returns, 0, None)
            if raw.sum() == 0:
                weights = np.repeat(1/len(raw), len(raw))
            else:
                weights = raw / raw.sum()
    return weights

ai_weights = ai_optimize_weights(data[selected_stocks], risk_profile)
ai_index = (norm_prices[selected_stocks] * ai_weights).sum(axis=1)

# --- INDEX CALCULATION ---
si = (norm_prices[selected_stocks] * weights).sum(axis=1)
sp500 = norm_prices['^GSPC']
ratio = si / sp500

# --- MARKET VALUATION CONTEXT ---
@st.cache_data(show_spinner=False)
def get_pe_ratios(tickers):
    pes = []
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            pes.append(info.get('trailingPE', np.nan))
        except Exception:
            pes.append(np.nan)
    return pd.Series(pes, index=tickers)

pe_ratios = get_pe_ratios(selected_stocks)

# --- PEER COMPARISON ---
peer_profiles = {
    "Conservative": np.repeat(1/len(selected_stocks), len(selected_stocks)),
    "Moderate": ai_optimize_weights(data[selected_stocks], "Moderate"),
    "Aggressive": ai_optimize_weights(data[selected_stocks], "Aggressive")
}
peer_returns = {k: ((norm_prices[selected_stocks] * v).sum(axis=1).iloc[-1] / (norm_prices[selected_stocks] * v).sum(axis=1).iloc[0] - 1) for k, v in peer_profiles.items()}

# --- DETAILED, ADVISOR-LIKE FEEDBACK ENGINE ---
def generate_feedback(selected_stocks, weights, risk_profile, pe_ratios, user_return, ai_return, ai_weights, weight_method, investment_goal, peer_returns):
    feedback = []
    top_stock = selected_stocks[np.argmax(weights)]
    feedback.append(
        f"<b>Portfolio Summary:</b><br>"
        f"- <b>Stocks Chosen:</b> {', '.join(selected_stocks)}<br>"
        f"- <b>Largest Weight:</b> {top_stock} ({weights[np.argmax(weights)]:.0%})<br>"
        f"- <b>Risk Profile:</b> {risk_profile}<br>"
        f"- <b>Investment Goal:</b> {investment_goal}"
    )
    if pe_ratios.notna().any():
        overvalued = pe_ratios[pe_ratios > 40]
        undervalued = pe_ratios[pe_ratios < 20]
        if not overvalued.empty:
            feedback.append(f"⚠️ <b>High P/E Alert:</b> {', '.join(overvalued.index)} have high P/E ratios. Consider if these fit your risk and goal.")
        if not undervalued.empty:
            feedback.append(f"🟢 <b>Value Alert:</b> {', '.join(undervalued.index)} have lower P/E ratios, which may offer value opportunities.")
    if np.max(weights) > 0.5:
        feedback.append("⚠️ <b>Concentration Risk:</b> More than 50% in one stock. This can lead to high volatility and risk. Diversification is generally safer.")
    elif np.max(weights) < 0.3 and len(selected_stocks) > 2:
        feedback.append("✅ <b>Good Diversification:</b> No single stock dominates your portfolio. This can help smooth returns and reduce risk.")
    if investment_goal == "Capital Preservation" and risk_profile != "Conservative":
        feedback.append("⚠️ You selected 'Capital Preservation' but a non-conservative risk profile. Consider aligning these for consistency.")
    if investment_goal == "Aggressive Growth" and risk_profile == "Conservative":
        feedback.append("⚠️ Aggressive growth goals may not be realistic with a conservative risk profile. Consider increasing risk tolerance or adjusting goals.")
    if user_return < ai_return:
        feedback.append(
            f"💡 <b>AI Suggestion:</b> Based on recent market data and your risk profile, our AI recommends an alternative allocation "
            f"(see below) that would have produced a higher return (<b>{ai_return*100:.2f}%</b>) than your allocation (<b>{user_return*100:.2f}%</b>)."
        )
        feedback.append(
            "<b>Alternative:</b> Try using the AI-optimized weights for a more data-driven approach. "
            "You can also experiment with the <b>What If?</b> scenarios below."
        )
    elif user_return > ai_return:
        feedback.append("✅ <b>Great job!</b> Your allocation outperformed the AI-optimized approach for this period.")
    else:
        feedback.append("ℹ️ Your portfolio performed similarly to the AI-optimized approach.")
    peer_msgs = []
    for peer, ret in peer_returns.items():
        peer_msgs.append(f"{peer}: {ret*100:.2f}%")
    feedback.append(f"<b>Peer Comparison:</b> Typical {', '.join(peer_msgs)} returns for these stocks and risk profiles.")
    feedback.append(
        "🧠 <b>Portfolio Tip:</b> A strong foundation is built on diversification, risk awareness, and adapting to market conditions. "
        "Review your allocations regularly and align them with your long-term goals."
    )
    return feedback

user_return = (si.iloc[-1] / si.iloc[0]) - 1
ai_return = (ai_index.iloc[-1] / ai_index.iloc[0]) - 1
feedback = generate_feedback(
    selected_stocks, weights, risk_profile, pe_ratios, user_return, ai_return, ai_weights, weight_method, investment_goal, peer_returns
)

# --- WHAT IF SCENARIO: Toggle between your weights and AI/peer alternatives ---
st.markdown('<div class="section-header">What If? Scenario Explorer</div>', unsafe_allow_html=True)
scenario = st.radio("See how your index would perform if you used:", 
    ["Your Current Weights", "AI-Optimized Weights", "Conservative Peer", "Aggressive Peer"])
if scenario == "Your Current Weights":
    scenario_index = si
    scenario_label = "Your Index"
elif scenario == "AI-Optimized Weights":
    scenario_index = ai_index
    scenario_label = "AI-Optimized"
elif scenario == "Conservative Peer":
    scenario_index = (norm_prices[selected_stocks] * peer_profiles["Conservative"]).sum(axis=1)
    scenario_label = "Conservative Peer"
else:
    scenario_index = (norm_prices[selected_stocks] * peer_profiles["Aggressive"]).sum(axis=1)
    scenario_label = "Aggressive Peer"

# --- METRICS DASHBOARD ---
st.markdown('<div class="section-header">Portfolio Metrics</div>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
col1.metric("Your Index Return", f"{user_return*100:.2f}%", delta=None)
col2.metric("AI-Optimized Return", f"{ai_return*100:.2f}%", delta=None)
col3.metric("S&P 500 Return", f"{(sp500.iloc[-1]/sp500.iloc[0] - 1)*100:.2f}%")

# --- VISUALIZATION ---
st.markdown(f'<div class="section-header">Performance Comparison: {scenario_label}</div>', unsafe_allow_html=True)
fig = go.Figure()
fig.add_trace(go.Scatter(x=scenario_index.index, y=scenario_index, name=scenario_label, line=dict(width=3, color='#1a73e8')))
fig.add_trace(go.Scatter(x=sp500.index, y=sp500, name="S&P 500", line=dict(width=2, dash='dash', color='#34a853')))
fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Normalized Value (Base=100)",
    height=450,
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig, use_container_width=True)

st.markdown('<div class="section-header">Index Ratio (Your Index / S&P 500)</div>', unsafe_allow_html=True)
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=ratio.index, y=ratio, name="Index Ratio", line=dict(width=2, color='#fbbf24')))
fig2.update_layout(
    xaxis_title="Date",
    yaxis_title="Ratio",
    height=250,
    hovermode="x unified"
)
st.plotly_chart(fig2, use_container_width=True)

# --- DETAILED ADVISOR FEEDBACK ---
st.markdown('<div class="section-header">AI Advisor’s Notes</div>', unsafe_allow_html=True)
for msg in feedback:
    st.markdown(f'<div class="advisor-feedback">{msg}</div>', unsafe_allow_html=True)

# --- ALTERNATIVE SUGGESTION (AI weights) ---
if user_return < ai_return:
    st.markdown('<div class="alt-suggestion"><b>Alternative Portfolio Suggestion:</b><br>'
                'Try these AI-optimized weights for your chosen stocks:<br>' +
                '<br>'.join([f"{stock}: <b>{w:.2f}</b>" for stock, w in zip(selected_stocks, ai_weights)]) +
                '</div>', unsafe_allow_html=True)

# --- PEER FEEDBACK ---
st.markdown('<div class="peer-feedback"><b>How do you compare?</b><br>'
            'See how your portfolio stacks up versus typical peer allocations for your chosen risk profile.</div>', unsafe_allow_html=True)

# --- CHEAT SHEET / EDUCATION ---
with st.expander("📚 Step 2 Cheat Sheet: Portfolio Construction & Benchmarking", expanded=False):
    st.markdown("""
    - **Your Index (t)** = Σ (weight × normalized price of each stock)
    - **S&P 500 (t)** = normalized S&P 500 price (base = 100)
    - **Index Ratio** = Your Index / S&P 500 (above 1 = outperformance)
    - **Diversification**: Reduces risk by spreading across assets.
    - **Concentration Risk**: Too much in one stock can lead to big losses.
    - **P/E Ratio**: High = potentially overvalued, Low = potentially undervalued.
    - **Align your risk profile and goals** for best results.
    - [Watch: Portfolio Construction Explained (YouTube)](https://www.youtube.com/watch?v=F4g3zjR7pYU)
    """)

# --- FINAL EDUCATIONAL CONTEXT ---
st.markdown("""
---
> **Step 2 Foundation:**  
> You’ve built a custom index, compared it to the S&P 500, and received individualized, actionable feedback from an AI advisor.  
> Experiment with weights, scenarios, and peer strategies to deepen your understanding.  
> This is the foundation for advanced analytics and agentic investing in future steps.
""")

