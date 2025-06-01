import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Step 2: Agentic AI Portfolio Builder", layout="wide")

st.title("Step 2: Foundation — Build Your Custom Index with Agentic AI Guidance")
st.markdown("""
**Construct your own index from the Magnificent 7 tech stocks, choose your weights, and compare to the S&P 500.  
Our AI advisor analyzes your portfolio, current market valuations, and your risk profile—offering tailored feedback and optimization suggestions.**
---
""")

# --- USER PROFILE ---
st.sidebar.header("Investor Profile")
risk_profile = st.sidebar.selectbox(
    "Your Risk Tolerance",
    ["Conservative", "Moderate", "Aggressive"],
    help="AI will optimize and critique your portfolio based on this."
)

# --- STOCK SELECTION ---
magnificent_7 = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META', 'NVDA', 'TSLA']
selected_stocks = st.sidebar.multiselect(
    "Select 1-7 Tech Stocks (Magnificent 7)",
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
    method_desc = "Each stock has equal influence in your index."
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
    method_desc = "Your custom weights reflect your own views or strategy."

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

start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-12-31"))
if start_date >= end_date:
    st.sidebar.error("Start Date must be before End Date.")
    st.stop()

tickers = selected_stocks + ['^GSPC']
data = fetch_data(tickers, start_date, end_date)
if data.empty or data.isnull().all().all():
    st.error("No data found for the selected period. Try a different date range or stocks.")
    st.stop()

norm_prices = data.div(data.iloc[0]) * 100

# --- AI OPTIMIZATION: ML-BASED SUGGESTED WEIGHTS ---
def ai_optimize_weights(prices, risk_profile):
    lookback = min(252, len(prices))
    returns = prices.pct_change().fillna(0).iloc[-lookback:]
    X = returns.values
    y = (prices.iloc[-1] / prices.iloc[0] - 1).values  # total return over period

    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)
    importances = model.feature_importances_

    # Risk profile adjustment
    if risk_profile == "Conservative":
        importances = np.clip(importances, 0, 0.25)
    elif risk_profile == "Moderate":
        importances = np.clip(importances, 0, 0.5)
    optimal_weights = importances / importances.sum()
    return optimal_weights

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

# --- INDIVIDUALIZED FEEDBACK ENGINE ---
def generate_feedback(selected_stocks, weights, risk_profile, pe_ratios, user_return, ai_return, ai_weights, weight_method):
    feedback = []
    # Explain user selection
    top_stock = selected_stocks[np.argmax(weights)]
    feedback.append(
        f"**You selected:** {', '.join(selected_stocks)}.\n"
        f"Your largest weight is in **{top_stock}** ({weights[np.argmax(weights)]:.0%})."
    )
    # Risk profile context
    if risk_profile == "Conservative":
        feedback.append("You indicated a **conservative** risk profile, so a more balanced, diversified allocation is typically preferred.")
    elif risk_profile == "Aggressive":
        feedback.append("You indicated an **aggressive** risk profile, so higher weights in growth stocks or concentrated bets are expected.")
    else:
        feedback.append("You indicated a **moderate** risk profile, so a mix of growth and stability is appropriate.")

    # Market context
    if pe_ratios.notna().any():
        overvalued = pe_ratios[pe_ratios > 40]
        undervalued = pe_ratios[pe_ratios < 20]
        if not overvalued.empty:
            feedback.append(f"⚠️ **High P/E Alert:** {', '.join(overvalued.index)} currently have high P/E ratios, suggesting they may be overvalued.")
        if not undervalued.empty:
            feedback.append(f"🟢 **Value Alert:** {', '.join(undervalued.index)} have lower P/E ratios, which may indicate better value.")

    # Weighting advice
    if weight_method == "Manual" and np.max(weights) > 0.5:
        feedback.append("⚠️ **Concentration Risk:** More than 50% in one stock increases risk. Diversification can help smooth returns.")
    elif weight_method == "Manual" and np.max(weights) < 0.3 and len(selected_stocks) > 2:
        feedback.append("✅ **Good Diversification:** No single stock dominates your portfolio.")

    # Performance comparison
    if user_return < ai_return:
        feedback.append(
            f"💡 **AI Suggestion:** Based on recent market data and your risk profile, our AI recommends an alternative allocation "
            f"(shown below) that would have produced a higher return ({ai_return*100:.2f}%) than your allocation ({user_return*100:.2f}%)."
        )
        feedback.append(
            "AI-optimized weights are calculated using machine learning on recent price patterns and adjusted for your risk profile."
        )
        feedback.append(
            "Try using the AI-optimized weights for a more tailored, data-driven approach."
        )
    elif user_return > ai_return:
        feedback.append("✅ **Great job!** Your allocation outperformed the AI-optimized approach for this period.")
    else:
        feedback.append("ℹ️ Your portfolio performed similarly to the AI-optimized approach.")

    return feedback

user_return = (si.iloc[-1] / si.iloc[0]) - 1
ai_return = (ai_index.iloc[-1] / ai_index.iloc[0]) - 1
feedback = generate_feedback(
    selected_stocks, weights, risk_profile, pe_ratios, user_return, ai_return, ai_weights, weight_method
)

# --- VISUALIZATION ---
fig = go.Figure()
fig.add_trace(go.Scatter(x=si.index, y=si, name="Your Index", line=dict(width=3)))
fig.add_trace(go.Scatter(x=sp500.index, y=sp500, name="S&P 500", line=dict(width=2, dash='dash')))
fig.add_trace(go.Scatter(x=ai_index.index, y=ai_index, name="AI-Optimized", line=dict(width=2, dash='dot')))
fig.update_layout(
    title="Your Custom Index vs. S&P 500 vs. AI-Optimized",
    xaxis_title="Date",
    yaxis_title="Normalized Value (Base=100)",
    height=500,
    hovermode="x unified"
)
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=ratio.index, y=ratio, name="SI(t) / S&P500(t)", line=dict(width=2, color='purple')))
fig2.update_layout(
    title="Index Ratio: SI(t) / S&P 500(t)",
    xaxis_title="Date",
    yaxis_title="Ratio",
    height=300,
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)
st.plotly_chart(fig2, use_container_width=True)

# --- INDIVIDUALIZED EXPLANATION & FEEDBACK ---
st.markdown("""
---
### **Your Personalized AI Feedback**
""")
for msg in feedback:
    st.markdown(f"- {msg}")

st.markdown("""
---
### **What’s Happening in Step 2: Foundation?**
- **Portfolio Construction:** Select stocks and assign weights, or let the AI suggest weights based on your risk profile and current market performance.
- **Formula:**  
  Your Index (t) ≈ Σ ωᵢ ⋅ (Pᵢ(t)/Pᵢ(0)) × 100  
  S&P 500 (t) ≈ (S(t)/S(0)) × 100  
  Index Ratio = Your Index (t) / S&P 500 (t)
- **Benchmarking:** Compare your custom index to the S&P 500 to see if you’re outperforming or underperforming.
- **Agentic AI Feedback:**  
  Personalized, actionable, and educational guidance—unique to your portfolio, your choices, and the market right now.
""")
