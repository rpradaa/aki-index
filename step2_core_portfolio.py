import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Agentic Adaptive Index", layout="wide")

st.title("Craft Your Core Portfolio (ω & P) — Agentic AI Foundation")
st.markdown("""
### Step 2: Foundation — Build, Analyze, and Learn with Agentic AI Guidance

This tool lets you construct your own index from the **Magnificent 7** tech stocks, assign weights (manually or with AI guidance), and compare to the S&P 500.  
You'll get real-time, context-aware feedback and actionable insights—just like a portfolio manager with an AI co-pilot.
---
""")

# --- SIDEBAR: USER INPUTS ---
st.sidebar.header("Portfolio Builder")

start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-12-31"))
if start_date >= end_date:
    st.sidebar.error("Start Date must be before End Date.")
    st.stop()

magnificent_7 = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META', 'NVDA', 'TSLA']
selected_stocks = st.sidebar.multiselect(
    "Select 1-7 Tech Stocks (Magnificent 7)",
    options=magnificent_7,
    default=magnificent_7[:3]
)
if len(selected_stocks) == 0:
    st.warning("Please select at least one stock.")
    st.stop()

weight_method = st.sidebar.selectbox(
    "Choose Weighting Method",
    options=["Equal Weight", "Custom Weights", "AI-Guided Weights"]
)

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

# --- AGENTIC/AI GUIDANCE: SUGGESTED WEIGHTS BASED ON VOLATILITY ---
def ai_guided_weights(prices):
    vol = prices.pct_change().std()
    inv_vol = 1 / (vol + 1e-8)
    weights = inv_vol / inv_vol.sum()
    return weights.values

# --- USER WEIGHTS & AGENTIC FEEDBACK ---
if weight_method == "Equal Weight":
    weights = np.repeat(1/len(selected_stocks), len(selected_stocks))
    method_desc = "Equal weighting gives each stock the same influence, regardless of risk or past performance."
elif weight_method == "Custom Weights":
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
    method_desc = "Custom weights let you express your own views or strategies."
else:  # AI-Guided
    weights = ai_guided_weights(norm_prices[selected_stocks])
    st.sidebar.markdown("**AI-Guided Weights (Risk-Aware):**")
    for stock, w in zip(selected_stocks, weights):
        st.sidebar.write(f"{stock}: {w:.2f}")
    method_desc = (
        "AI-guided weights use a risk-aware approach: stocks with lower volatility get higher weights, "
        "helping you build a more stable portfolio. This is inspired by 'risk parity' strategies used by leading funds."
    )

# --- AGENTIC AI FEEDBACK & SUGGESTIONS ---
returns = norm_prices[selected_stocks].iloc[-1] / norm_prices[selected_stocks].iloc[0] - 1
vols = norm_prices[selected_stocks].pct_change().std() * np.sqrt(252)
portfolio_return = (norm_prices[selected_stocks] * weights).sum(axis=1)
total_return = (portfolio_return.iloc[-1] / portfolio_return.iloc[0]) - 1
annualized_vol = portfolio_return.pct_change().std() * np.sqrt(252)

# AI critiques and suggestions
ai_feedback = ""
ai_suggestion = ""
flag = False

if weight_method == "Custom Weights":
    if np.max(weights) > 0.6:
        ai_feedback += "⚠️ Your portfolio is highly concentrated in one stock. This increases risk and can lead to large swings in performance.\n\n"
        flag = True
    if np.min(weights) < 0.05 and len(selected_stocks) > 2:
        ai_feedback += "ℹ️ You have some stocks with very low weights. Consider if they add value or just add complexity.\n\n"
    # Compare to AI-guided
    ai_weights = ai_guided_weights(norm_prices[selected_stocks])
    ai_portfolio = (norm_prices[selected_stocks] * ai_weights).sum(axis=1)
    ai_total_return = (ai_portfolio.iloc[-1] / ai_portfolio.iloc[0]) - 1
    ai_annualized_vol = ai_portfolio.pct_change().std() * np.sqrt(252)
    if total_return < ai_total_return:
        ai_suggestion += f"💡 **AI Suggestion:** An AI-guided, risk-aware allocation would have achieved a higher return ({ai_total_return*100:.2f}%) with lower risk ({ai_annualized_vol*100:.2f}% volatility) over this period. Consider using AI-guided weights for a more balanced approach.\n"
    else:
        ai_suggestion += f"✅ Your custom allocation outperformed the AI-guided approach in this period. Nice work!\n"
    if flag:
        ai_suggestion += "Try spreading your weights more evenly or using the AI-guided method for more stability.\n"
elif weight_method == "Equal Weight":
    ai_weights = ai_guided_weights(norm_prices[selected_stocks])
    ai_portfolio = (norm_prices[selected_stocks] * ai_weights).sum(axis=1)
    ai_total_return = (ai_portfolio.iloc[-1] / ai_portfolio.iloc[0]) - 1
    ai_annualized_vol = ai_portfolio.pct_change().std() * np.sqrt(252)
    ai_feedback += f"ℹ️ Equal weighting is simple and robust. But a risk-aware, AI-guided allocation would have produced a return of {ai_total_return*100:.2f}% with volatility {ai_annualized_vol*100:.2f}% over this period. Try it to see the difference!"
else:
    ai_feedback += "✅ You are using AI-guided weights. This approach is designed to reduce risk by allocating more to less volatile stocks, a method used by many professional investors."

# Diversification check
if np.max(weights) > 0.6:
    st.warning("⚠️ Your portfolio is highly concentrated in one stock. Consider diversifying for more stability (AI best practice).")
elif np.max(weights) < 0.4 and len(selected_stocks) > 1:
    st.success("✅ Good diversification: No single stock dominates your portfolio.")

st.markdown(f"**AI Advisor Tip:** {method_desc}")
st.markdown(ai_feedback)
st.markdown(ai_suggestion)

# --- Show Stock Stats ---
st.markdown("#### Quick Stock Stats")
stats = pd.DataFrame({
    "Total Return (%)": (returns * 100).round(2),
    "Annualized Volatility (%)": (vols * 100).round(2),
    "Weight": np.round(weights, 2)
}, index=selected_stocks)
st.dataframe(stats)

# --- CALCULATE INDEX AND RATIO ---
si = (norm_prices[selected_stocks] * weights).sum(axis=1)
sp500 = norm_prices['^GSPC']
ratio = si / sp500

# --- VISUALIZATION ---
fig = go.Figure()
fig.add_trace(go.Scatter(x=si.index, y=si, name="Your Index (SI)", line=dict(width=3)))
fig.add_trace(go.Scatter(x=sp500.index, y=sp500, name="S&P 500", line=dict(width=2, dash='dash')))
fig.update_layout(
    title="Your Custom Index vs. S&P 500",
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

# --- EXPLANATION & VALUE ---
st.markdown("""
---

### **Step 2: Foundation — What’s Happening?**
- **Portfolio Construction:** You select stocks and assign weights, or let the AI suggest weights based on recent volatility (risk-aware).
- **Agentic AI Guidance:** The system provides instant feedback on diversification, risk, and performance, and suggests improvements or alternatives.
- **Benchmarking:** The S&P 500 is your reference—see if your custom index outperforms or underperforms.
- **Educational:** Learn portfolio theory and best practices interactively, with AI feedback.

#### **What is the Index Ratio?**
- The **Index Ratio** = Your Index (t) / S&P 500 (t)
- **Above 1:** Outperforming S&P 500. **Below 1:** Underperforming.
- This gives you a clear, at-a-glance sense of relative performance.

#### **Agentic Value**
- **Responsive:** The AI adapts its feedback and suggestions to your choices, helping you learn and improve.
- **Actionable:** Use the platform to explore, optimize, and understand portfolio construction at a professional level.

---

**Formula:**  
Your Index (t) ≈ Σ ωᵢ ⋅ (Pᵢ(t)/Pᵢ(0)) × 100  
S&P 500 (t) ≈ (S(t)/S(0)) × 100  
Index Ratio = Your Index (t) / S&P 500 (t)
""")
