import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- CONFIG ---
st.set_page_config(page_title="Adaptive Index Foundation", layout="wide")

# --- HEADER ---
st.title("Craft Your Core Portfolio (ω & P)")
st.markdown("""
**Step 2: Foundation**  
Build your own custom index by selecting stocks, weights, and comparing to the S&P 500.
""")

# --- USER INPUTS ---
st.sidebar.header("Portfolio Builder")

# 1. Date Range
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-12-31"))

# 2. Stock Selection
tech_stocks = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META', 'NVDA', 'TSLA']
selected_stocks = st.sidebar.multiselect(
    "Select 3-5 Tech Stocks",
    options=tech_stocks,
    default=tech_stocks[:3]
)
if not (3 <= len(selected_stocks) <= 5):
    st.warning("Please select between 3 and 5 stocks.")
    st.stop()

# 3. Weighting Method
weight_method = st.sidebar.selectbox(
    "Choose Weighting Method",
    options=["Equal Weight", "Custom Weights"]
)

if weight_method == "Equal Weight":
    weights = np.repeat(1/len(selected_stocks), len(selected_stocks))
else:
    weights = []
    st.sidebar.markdown("Enter custom weights (must sum to 1):")
    for stock in selected_stocks:
        w = st.sidebar.number_input(f"Weight for {stock}", min_value=0.0, max_value=1.0, value=1/len(selected_stocks), step=0.01)
        weights.append(w)
    weights = np.array(weights)
    if not np.isclose(weights.sum(), 1.0):
        st.warning("Custom weights must sum to 1. Adjust your weights.")
        st.stop()

# --- DATA FETCHING ---
@st.cache_data(show_spinner=True)
def fetch_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end)['Adj Close']
    return data

tickers = selected_stocks + ['^GSPC']
data = fetch_data(tickers, start_date, end_date)

if data.empty or data.isnull().all().all():
    st.error("No data found for the selected period. Try a different date range.")
    st.stop()

# --- NORMALIZATION ---
norm_prices = data.div(data.iloc[0]) * 100
si = (norm_prices[selected_stocks] * weights).sum(axis=1)
sp500 = norm_prices['^GSPC']
ratio = si / sp500

# --- VISUALIZATION ---
fig = go.Figure()
fig.add_trace(go.Scatter(x=si.index, y=si, name="Your Index (SI)", line=dict(width=3)))
fig.add_trace(go.Scatter(x=sp500.index, y=sp500, name="S&P 500", line=dict(width=2, dash='dash')))
fig.update_layout(title="Your Custom Index vs. S&P 500", xaxis_title="Date", yaxis_title="Normalized Value (Base=100)", height=500)

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=ratio.index, y=ratio, name="SI(t) / S&P500(t)", line=dict(width=2, color='purple')))
fig2.update_layout(title="Index Ratio: SI(t) / S&P 500(t)", xaxis_title="Date", yaxis_title="Ratio", height=300)

st.plotly_chart(fig, use_container_width=True)
st.plotly_chart(fig2, use_container_width=True)

# --- FORMULA SNIPPET ---
st.markdown("""
**Formula:**  
Your Index (t) ≈ Σ ωᵢ ⋅ (Pᵢ(t)/Pᵢ(0)) × 100  
S&P 500 (t) ≈ (S(t)/S(0)) × 100  
Ratio = Your Index (t) / S&P 500 (t)
""")

# --- TAKEAWAY ---
st.info("You've built a basic custom index. You can see how your chosen weights affect its performance relative to the S&P 500 based purely on price.")

