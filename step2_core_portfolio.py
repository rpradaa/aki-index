# In your streamlit_app.py
import streamlit as st
import pandas as pd
from step2_core_portfolio import (
    get_stock_data_step2, 
    robust_normalize_prices_step2, 
    calculate_simple_weighted_index_step2
    # No need to import plot_step2_results if you re-implement plotting in Streamlit
)
import plotly.graph_objects as go # For st.plotly_chart

st.title("AlphaKnaut - Step 2: Core Portfolio")

# --- Streamlit UI elements for inputs ---
selected_stocks = st.multiselect("Select Stocks:", ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'AMZN', 'TSLA'], default=['AAPL', 'MSFT'])
start_date = st.date_input("Start Date", pd.to_datetime("2023-01-01"))
end_date = st.date_input("End Date", pd.Timestamp.today() - pd.Timedelta(days=1))

# Weighting choice
weight_option = st.radio("Weighting Method (ω):", ("Equal Weights", "Custom Weights"))

final_weights = {}
if weight_option == "Custom Weights":
    custom_weights_valid = True
    total_weight = 0
    for stock in selected_stocks:
        weight = st.number_input(f"Weight for {stock} (0.0 to 1.0):", min_value=0.0, max_value=1.0, value=1.0/len(selected_stocks) if selected_stocks else 0.0, step=0.01, key=f"weight_{stock}")
        final_weights[stock] = weight
        total_weight += weight
    if selected_stocks and not np.isclose(total_weight, 1.0):
        st.warning(f"Custom weights sum to {total_weight:.2f}, not 1.0. Please adjust for accurate representation or they will be normalized.")
        if st.button("Normalize My Custom Weights"):
            if total_weight > 0 :
                for stock in final_weights: final_weights[stock] /= total_weight
                st.success(f"Weights normalized: {final_weights}")
                # Streamlit reruns on button click, values will be updated in next run
            else:
                st.error("Cannot normalize weights if sum is zero.")


elif selected_stocks: # Equal weights
    final_weights = {ticker: 1.0 / len(selected_stocks) for ticker in selected_stocks}

if st.button("Analyze Core Portfolio"):
    if not selected_stocks:
        st.warning("Please select at least one stock.")
    elif not final_weights:
         st.warning("Weights are not defined. Please configure weights.")
    else:
        st.write(f"Using Base Weights (ω): {final_weights}")
        
        tickers_to_fetch = list(final_weights.keys()) + ['^GSPC']
        all_price_data = get_stock_data_step2(tickers_to_fetch, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

        if not all_price_data.empty:
            # ... (rest of the processing: normalize, calculate index) ...
            normalized_price_data = robust_normalize_prices_step2(all_price_data)
            if not normalized_price_data.empty and not normalized_price_data.isnull().all().all():
                valid_components = [s for s in final_weights.keys() if s in normalized_price_data.columns]
                user_portfolio_components_normalized = normalized_price_data[valid_components]
                benchmark_normalized = normalized_price_data.get('^GSPC')

                if not user_portfolio_components_normalized.empty:
                    user_simple_index = calculate_simple_weighted_index_step2(user_portfolio_components_normalized, final_weights)
                    
                    if not user_simple_index.empty:
                        # Plotting in Streamlit
                        fig1 = go.Figure()
                        fig1.add_trace(go.Scatter(x=user_simple_index.index, y=user_simple_index, name="Your Core Portfolio"))
                        if benchmark_normalized is not None and not benchmark_normalized.empty:
                            fig1.add_trace(go.Scatter(x=benchmark_normalized.index, y=benchmark_normalized, name="S&P 500 (Norm)", line=dict(dash='dash')))
                        fig1.update_layout(title="Core Portfolio vs. S&P 500", yaxis_title="Normalized Value")
                        st.plotly_chart(fig1)

                        # Ratio plot
                        if benchmark_normalized is not None and not benchmark_normalized.empty:
                            aligned_user_index, aligned_benchmark = user_simple_index.align(benchmark_normalized, join='inner')
                            if not aligned_user_index.empty:
                                safe_aligned_benchmark = aligned_benchmark.replace(0, np.nan)
                                ratio_series = aligned_user_index / safe_aligned_benchmark
                                ratio_series.dropna(inplace=True)
                                if not ratio_series.empty:
                                    fig2 = go.Figure()
                                    fig2.add_trace(go.Scatter(x=ratio_series.index, y=ratio_series, name="Portfolio / S&P 500 Ratio"))
                                    fig2.add_hline(y=1, line_dash="dash")
                                    fig2.update_layout(title="Performance Ratio")
                                    st.plotly_chart(fig2)
                    else: st.error("Failed to calculate custom index.")
                else: st.error("No valid component data after normalization.")
            else: st.error("Normalization failed or resulted in all NaNs.")
        else: st.error("Failed to fetch price data.")
