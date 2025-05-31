import yfinance as yf
import pandas as pd
import numpy as np # For np.isclose
import plotly.graph_objects as go
# from plotly.offline import plot # To save as HTML file if not in interactive environment
# For direct plotting in some IDEs or if notebook mode is configured:
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True) # Keep for environments that support it, harmless otherwise

# --- Configuration ---
DEFAULT_BENCHMARK_TICKER = '^GSPC'
DEFAULT_BENCHMARK_NAME = 'S&P 500'

# --- Core Data Functions (Simplified for Step 2) ---

def get_stock_data_step2(tickers_list, start_date_str, end_date_str):
    """
    Downloads 'Adj Close' (via auto_adjust=True) prices for selected tickers.
    Returns a DataFrame with dates as index and tickers as columns.
    """
    if not tickers_list:
        print("Error: No tickers provided for data download.")
        return pd.DataFrame()
    
    print(f"\nFetching data for: {', '.join(tickers_list)} from {start_date_str} to {end_date_str}...")
    try:
        # auto_adjust=True directly gives adjusted prices, often under 'Close' or just ticker name
        data = yf.download(tickers_list, start=start_date_str, end=end_date_str, 
                           auto_adjust=True, progress=False, timeout=20)
        
        if data.empty:
            print("Error: No data downloaded from yfinance.")
            return pd.DataFrame()

        # If multiple tickers, yf might return a multi-index header for columns (e.g., 'Close', 'Open').
        # If single ticker, it might be a simple DataFrame or Series.
        # We are interested in the 'Close' prices (which are adjusted due to auto_adjust=True).
        
        if isinstance(data.columns, pd.MultiIndex):
            # If 'Close' is a level, use it. Otherwise, try to infer or use the first level.
            if 'Close' in data.columns.levels[0]:
                price_data = data['Close']
            else: # Fallback if 'Close' isn't the primary level (e.g. if only one type like 'Adj Close' was there)
                print("Warning: 'Close' not found as primary level in MultiIndex columns. Using first available price type.")
                price_data = data.iloc[:, data.columns.get_level_values(1).isin(tickers_list)] # More robust selection
                # This might need more refinement based on exact yfinance output for specific cases
                if isinstance(price_data.columns, pd.MultiIndex): # If still multi-index after selection
                    price_data = price_data.droplevel(0, axis=1) # Drop the price type level
        elif len(tickers_list) == 1 and isinstance(data, pd.DataFrame) and 'Close' in data.columns : # Single ticker, DataFrame
             price_data = data[['Close']]
             price_data.columns = [tickers_list[0]] # Ensure column name is the ticker
        elif len(tickers_list) == 1 and isinstance(data, pd.Series) : # Single ticker, yf returns Series
            price_data = data.to_frame(name=tickers_list[0])
        elif all(ticker in data.columns for ticker in tickers_list): # Multiple tickers, simple columns
            price_data = data[tickers_list]
        else:
            print("Error: Could not properly extract 'Close' prices. Data structure unexpected.")
            print("Data columns:", data.columns)
            return pd.DataFrame()

        return price_data.ffill().bfill().dropna(how='all')
    
    except Exception as e:
        print(f"Error during data download for Step 2: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def robust_normalize_prices_step2(price_data_df):
    """
    Normalizes prices to a base of 100.
    Handles potential NaNs or zeros in the first row used for normalization.
    """
    if price_data_df is None or price_data_df.empty:
        print("Error: Price data is empty for normalization.")
        return pd.DataFrame()

    normalized_cols = {}
    for col in price_data_df.columns:
        col_series = price_data_df[col].copy() # Work on a copy of the column
        first_valid_value_for_norm = None

        # Try to find the first non-NaN, non-zero value
        temp_filled_series = col_series.ffill().bfill() # Fill NaNs to find first value
        if not temp_filled_series.empty:
            for val in temp_filled_series:
                if pd.notna(val) and val != 0:
                    first_valid_value_for_norm = val
                    break
        
        if first_valid_value_for_norm is not None:
            normalized_cols[col] = (col_series / first_valid_value_for_norm) * 100
        else:
            print(f"Warning: Could not find a valid non-zero first value for normalization in column '{col}'. Column will contain NaNs or original values scaled by 100.")
            # Fallback: if no valid normalizer, scale by 100 to avoid div by zero, or return NaNs
            # This indicates data quality issues for this column for the given period.
            normalized_cols[col] = col_series * 100 # Or pd.Series(np.nan, index=col_series.index)
            
    normalized_df = pd.DataFrame(normalized_cols)
    # After normalization, the first valid data point for each series should be 100.
    # We still ffill/bfill to handle initial NaNs if the series started late.
    return normalized_df.ffill().bfill()


def calculate_simple_weighted_index_step2(normalized_stock_prices_df, weights_dict):
    """
    Calculates a simple weighted index: Σ (ω_i * NormalizedPrice_i).
    Assumes NormalizedPrice_i is already P'(t) (i.e., (P(t)/P(0))*100).
    Weights should sum to 1.0.
    """
    if normalized_stock_prices_df.empty or not weights_dict:
        print("Error: Cannot calculate simple index with empty prices or weights.")
        return pd.Series(dtype='float64')

    # Ensure weights dictionary only contains tickers present in the price data
    valid_weights = {ticker: weight for ticker, weight in weights_dict.items() if ticker in normalized_stock_prices_df.columns}
    if not valid_weights:
        print("Error: No valid weights for available stock tickers.")
        return pd.Series(dtype='float64')

    # Optional: Re-normalize valid_weights if they don't sum to 1 (e.g., if some tickers were dropped)
    # current_weight_sum = sum(valid_weights.values())
    # if not np.isclose(current_weight_sum, 1.0) and current_weight_sum != 0:
    #     print(f"Adjusting weights to sum to 1 from {current_weight_sum:.3f}")
    #     valid_weights = {ticker: weight / current_weight_sum for ticker, weight in valid_weights.items()}

    # Select only the components for which we have valid weights
    index_components_df = normalized_stock_prices_df[list(valid_weights.keys())]
    component_weights_series = pd.Series(valid_weights)

    # Perform weighted sum: Component_Normalized_Price * Weight_Component
    # This results in a DataFrame where each column is the weighted contribution of that stock to the index
    weighted_components_df = index_components_df.multiply(component_weights_series, axis='columns')
    
    # Sum these weighted contributions across all stocks for each day
    simple_index_series = weighted_components_df.sum(axis=1)
    
    return simple_index_series

# --- Plotting Function ---
def plot_step2_results(user_index_series, benchmark_normalized_series, 
                       user_index_name="Your Custom Portfolio", benchmark_name=DEFAULT_BENCHMARK_NAME,
                       filename_prefix="step2_plot"):
    """Plots the user's simple index vs. benchmark and their ratio."""
    if user_index_series.empty:
        print("Cannot plot: User index series is empty.")
        return

    # Plot 1: Performance Comparison
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=user_index_series.index, y=user_index_series,
                              mode='lines', name=user_index_name, line=dict(width=2)))
    if not benchmark_normalized_series.empty:
        fig1.add_trace(go.Scatter(x=benchmark_normalized_series.index, y=benchmark_normalized_series,
                                  mode='lines', name=benchmark_name + " (Normalized)",
                                  line=dict(dash='dash', color='grey')))
    fig1.update_layout(title=f"Step 2: {user_index_name} vs. {benchmark_name}",
                       xaxis_title="Date", yaxis_title="Normalized Value (Base 100)",
                       legend_title_text="Index")
    
    # In a script, you might want to save to HTML or show if environment supports it
    # plot(fig1, filename=f'{filename_prefix}_performance.html', auto_open=False) 
    print(f"\nDisplaying Performance Plot (if in suitable environment, or check HTML files)...")
    iplot(fig1)


    # Plot 2: Ratio Plot (User Index / Benchmark)
    if not benchmark_normalized_series.empty:
        # Ensure alignment for ratio calculation
        aligned_user_index, aligned_benchmark = user_index_series.align(benchmark_normalized_series, join='inner')
        
        if not aligned_user_index.empty and not aligned_benchmark.empty:
            # Avoid division by zero if benchmark somehow became zero (unlikely for normalized data > 0)
            safe_aligned_benchmark = aligned_benchmark.replace(0, np.nan) # Replace 0 with NaN
            ratio_series = aligned_user_index / safe_aligned_benchmark
            ratio_series.dropna(inplace=True) # Remove NaNs resulting from division by NaN

            if not ratio_series.empty:
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=ratio_series.index, y=ratio_series, mode='lines',
                                          name=f"{user_index_name} / {benchmark_name} Ratio",
                                          line=dict(color='purple')))
                fig2.add_hline(y=1, line_dash="dash", line_color="black", 
                               annotation_text="Ratio = 1 (Equal Relative Performance)", 
                               annotation_position="bottom right")
                fig2.update_layout(title=f"Performance Ratio: {user_index_name} vs. {benchmark_name}",
                                   xaxis_title="Date", yaxis_title="Ratio")
                # plot(fig2, filename=f'{filename_prefix}_ratio.html', auto_open=False)
                print(f"\nDisplaying Ratio Plot...")
                iplot(fig2)
            else:
                print("Could not calculate or plot the performance ratio (e.g., no overlapping data or benchmark was zero).")
        else:
            print("Could not align custom index and benchmark for ratio calculation.")
    else:
        print("Benchmark data not available for ratio plot.")


# --- Main Execution Block for Step 2 ---
def run_foundational_portfolio_analysis(selected_stocks, base_weights_input, start_date, end_date):
    """
    Orchestrates the analysis for Step 2.
    base_weights_input can be 'equal' or a dictionary like {'AAPL': 0.5, 'MSFT': 0.5}
    """
    print("--- AlphaKnaut: Step 2 - Crafting Your Core Portfolio ---")
    
    if not selected_stocks:
        print("Error: Please provide a list of stocks to analyze.")
        return

    # 1. Determine Weights (ω)
    final_weights = {}
    if isinstance(base_weights_input, str) and base_weights_input.lower() == 'equal':
        if selected_stocks:
            final_weights = {ticker: 1.0 / len(selected_stocks) for ticker in selected_stocks}
        else:
            print("Error: Cannot use 'equal' weights with no selected stocks.")
            return
    elif isinstance(base_weights_input, dict):
        # Validate custom weights sum to 1 (approximately)
        current_sum = sum(base_weights_input.values())
        if not np.isclose(current_sum, 1.0):
            print(f"Warning: Provided custom weights sum to {current_sum:.3f}, not 1.0.")
            # Option 1: Normalize them
            if current_sum != 0:
                print("Normalizing weights to sum to 1.0...")
                final_weights = {ticker: weight / current_sum for ticker, weight in base_weights_input.items()}
            else:
                print("Error: Custom weights sum to 0. Cannot normalize. Using equal weights as fallback.")
                final_weights = {ticker: 1.0 / len(selected_stocks) for ticker in selected_stocks}
            # Option 2: Raise an error or proceed with unnormalized weights (less ideal for comparison)
        else:
            final_weights = base_weights_input
    else:
        print("Error: Invalid 'base_weights_input'. Please provide 'equal' or a dictionary of weights.")
        return
        
    print(f"Using Base Weights (ω): {final_weights}")

    # 2. Get Data
    tickers_to_fetch = list(final_weights.keys()) + [DEFAULT_BENCHMARK_TICKER] # Ensure benchmark is fetched
    all_price_data = get_stock_data_step2(tickers_to_fetch, start_date, end_date)

    if all_price_data.empty:
        print("Failed to fetch price data. Aborting Step 2 analysis.")
        return
        
    # Ensure all selected stocks and benchmark are in the fetched data
    missing_assets = [t for t in tickers_to_fetch if t not in all_price_data.columns]
    if missing_assets:
        print(f"Error: Data missing for the following assets after download: {', '.join(missing_assets)}. Aborting.")
        return

    # 3. Normalize Prices
    normalized_price_data = robust_normalize_prices_step2(all_price_data)
    if normalized_price_data.empty:
        print("Failed to normalize price data. Aborting Step 2 analysis.")
        return

    user_portfolio_components_normalized = normalized_price_data[list(final_weights.keys())]
    benchmark_normalized = normalized_price_data[DEFAULT_BENCHMARK_TICKER]

    # 4. Calculate Simple Weighted Index
    user_simple_index = calculate_simple_weighted_index_step2(user_portfolio_components_normalized, final_weights)
    if user_simple_index.empty:
        print("Failed to calculate your custom simple index. Aborting Step 2 analysis.")
        return

    # 5. Plot Results
    plot_step2_results(user_simple_index, benchmark_normalized, 
                       user_index_name="Your Core Portfolio (ω & P)", 
                       benchmark_name=DEFAULT_BENCHMARK_NAME)
    
    print("\n--- Step 2 Analysis Complete ---")


if __name__ == "__main__":
    # This block runs if you execute the script directly (e.g., python step2_core_portfolio.py)
    
    # --- User Inputs (Simulated for script execution) ---
    # Scenario 1: Equal Weights
    print("\n\n--- SCENARIO 1: EQUAL WEIGHTS ---")
    user_selected_stocks_eq = ['AAPL', 'MSFT', 'GOOGL', 'NVDA']
    user_weights_eq = 'equal' # Special string to trigger equal weighting
    start_date_input_eq = "2023-01-01"
    end_date_input_eq = pd.Timestamp.today().strftime('%Y-%m-%d') # Use today as end date for example

    run_foundational_portfolio_analysis(
        selected_stocks=user_selected_stocks_eq,
        base_weights_input=user_weights_eq,
        start_date=start_date_input_eq,
        end_date=end_date_input_eq
    )

    # Scenario 2: Custom Weights
    print("\n\n--- SCENARIO 2: CUSTOM WEIGHTS ---")
    user_selected_stocks_custom = ['AAPL', 'AMZN', 'TSLA']
    user_weights_custom = {'AAPL': 0.5, 'AMZN': 0.3, 'TSLA': 0.2} # Weights sum to 1.0
    start_date_input_custom = "2023-06-01"
    end_date_input_custom = pd.Timestamp.today().strftime('%Y-%m-%d')

    run_foundational_portfolio_analysis(
        selected_stocks=user_selected_stocks_custom,
        base_weights_input=user_weights_custom,
        start_date=start_date_input_custom,
        end_date=end_date_input_custom
    )

    # Scenario 3: Custom Weights that don't sum to 1 (to test normalization)
    print("\n\n--- SCENARIO 3: CUSTOM WEIGHTS (NON-NORMALIZED INPUT) ---")
    user_selected_stocks_custom_nn = ['GOOGL', 'META']
    user_weights_custom_nn = {'GOOGL': 0.7, 'META': 0.5} # Sums to 1.2
    
    run_foundational_portfolio_analysis(
        selected_stocks=user_selected_stocks_custom_nn,
        base_weights_input=user_weights_custom_nn,
        start_date=start_date_input_custom, # Re-use dates
        end_date=end_date_input_custom
    )
