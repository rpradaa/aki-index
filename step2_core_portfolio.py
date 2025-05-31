# step2_core_portfolio.py - V2 (Corrected for ImportError)

import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.offline import plot # Use this for saving HTML files

# --- Configuration ---
DEFAULT_BENCHMARK_TICKER = '^GSPC'
DEFAULT_BENCHMARK_NAME = 'S&P 500'

# --- Core Data Functions ---
def get_stock_data_step2(tickers_list, start_date_str, end_date_str):
    if not tickers_list:
        print("Error: No tickers provided for data download.")
        return pd.DataFrame()
    print(f"\nFetching data for: {', '.join(tickers_list)} from {start_date_str} to {end_date_str}...")
    try:
        # auto_adjust=True usually simplifies to 'Close' being the adjusted price
        data = yf.download(tickers_list, start=start_date_str, end=end_date_str, 
                           auto_adjust=True, progress=False, timeout=20)
        
        if data.empty:
            print("Error: No data downloaded from yfinance.")
            return pd.DataFrame()
        
        price_data = pd.DataFrame()
        # Handle different structures yfinance might return with auto_adjust=True
        if isinstance(data.columns, pd.MultiIndex):
            # Common case: ('Close', 'AAPL'), ('Open', 'AAPL'), etc.
            # We want the 'Close' part.
            if 'Close' in data.columns.levels[0]:
                price_data = data['Close']
            elif len(data.columns.levels[0]) > 0 : # If no 'Close', take the first price type (e.g. 'Adj Close' if it was named differently)
                price_type_to_use = data.columns.levels[0][0]
                print(f"Warning: 'Close' not found as primary level in MultiIndex. Using '{price_type_to_use}'.")
                price_data = data[price_type_to_use]
            else: # Unrecognized MultiIndex structure
                print(f"Error: Unrecognized MultiIndex column structure from yfinance: {data.columns}")
                return pd.DataFrame()
            
            # Ensure price_data has the tickers as columns if it was extracted correctly
            # If after selecting price_type, we still have a MultiIndex (e.g. only one price type was present for all)
            if isinstance(price_data.columns, pd.MultiIndex):
                price_data = price_data.droplevel(0, axis=1) # Drop the price type level

            # Filter down to only the requested tickers if more were somehow returned (unlikely for 'Close' selection)
            price_data = price_data[[col for col in tickers_list if col in price_data.columns]]


        elif len(tickers_list) == 1: # Single ticker, yfinance might return DataFrame or Series
            if isinstance(data, pd.DataFrame):
                if 'Close' in data.columns:
                    price_data = data[['Close']] # Keep as DataFrame
                elif not data.empty: # If no 'Close' column, assume the DataFrame is the price data itself
                    price_data = data 
                else:
                    print(f"Error: Single ticker {tickers_list[0]} DataFrame empty or no 'Close' column.")
                    return pd.DataFrame()
                price_data.columns = [tickers_list[0]] # Ensure correct column name

            elif isinstance(data, pd.Series) : 
                price_data = data.to_frame(name=tickers_list[0])
            else:
                print(f"Error: Unexpected data type for single ticker {tickers_list[0]}: {type(data)}")
                return pd.DataFrame()
        
        elif all(ticker in data.columns for ticker in tickers_list): # Multiple tickers, simple columns (most common for auto_adjust=True)
            price_data = data[tickers_list]
        else: # Attempt to select by list of tickers if some exist, even if not all
            available_cols = [col for col in tickers_list if col in data.columns]
            if available_cols:
                print(f"Warning: Not all requested tickers found as direct columns. Using available: {available_cols}")
                price_data = data[available_cols]
            else:
                print(f"Error: Could not properly extract prices. Tickers not found in columns: {data.columns}")
                return pd.DataFrame()

        return price_data.ffill().bfill().dropna(how='all')
    except Exception as e:
        print(f"Error during data download for Step 2: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def robust_normalize_prices_step2(price_data_df):
    if price_data_df is None or price_data_df.empty:
        print("Error: Price data is empty for normalization.")
        return pd.DataFrame()
    normalized_cols = {}
    for col in price_data_df.columns:
        col_series = price_data_df[col].copy()
        first_valid_value_for_norm = None
        temp_filled_series = col_series.ffill().bfill() 
        if not temp_filled_series.empty:
            # Iterate to find the first non-NaN, non-zero value
            for val in temp_filled_series:
                if pd.notna(val) and val != 0:
                    first_valid_value_for_norm = val
                    break # Found it
        
        if first_valid_value_for_norm is not None:
            normalized_cols[col] = (col_series / first_valid_value_for_norm) * 100
        else:
            print(f"Warning: Could not find a valid non-zero first value for normalization in column '{col}'. This column will likely be all NaNs or scaled by 100 if original data had issues.")
            # If no valid normalizer, results for this column will be problematic.
            # Returning NaNs is safer than arbitrary scaling if data quality is poor.
            normalized_cols[col] = pd.Series(np.nan, index=col_series.index, name=col) 
            
    normalized_df = pd.DataFrame(normalized_cols)
    # ffill/bfill after all normalizations attempt to fill leading NaNs if series start at different times
    # but if a column was all NaN or all zero from start, it remains so.
    return normalized_df.ffill().bfill() 

def calculate_simple_weighted_index_step2(normalized_stock_prices_df, weights_dict):
    if normalized_stock_prices_df.empty or not weights_dict:
        print("Error: Cannot calculate simple index with empty prices or weights.")
        return pd.Series(dtype='float64')

    # Filter weights to only include tickers present in the normalized_stock_prices_df
    valid_weights = {ticker: weight for ticker, weight in weights_dict.items() 
                     if ticker in normalized_stock_prices_df.columns and pd.notna(normalized_stock_prices_df[ticker]).any()} # Ensure column has some non-NaN data
    
    if not valid_weights:
        print("Error: No valid weights for available stock tickers with data.")
        return pd.Series(dtype='float64')

    # Select only the components for which we have valid weights AND data
    index_components_df = normalized_stock_prices_df[list(valid_weights.keys())]
    
    # If after selecting valid keys, some columns in index_components_df are all NaN, they shouldn't contribute
    index_components_df = index_components_df.dropna(axis=1, how='all') # Drop columns that are all NaN
    
    # Re-filter valid_weights to match the actually available columns in index_components_df
    final_valid_weights = {ticker: weight for ticker, weight in valid_weights.items() if ticker in index_components_df.columns}
    if not final_valid_weights:
        print("Error: No components left after filtering all-NaN columns.")
        return pd.Series(dtype='float64')

    component_weights_series = pd.Series(final_valid_weights)
    
    # Ensure weights still sum to 1 after potential filtering, if not, re-normalize
    current_sum_of_final_weights = component_weights_series.sum()
    if not np.isclose(current_sum_of_final_weights, 1.0) and current_sum_of_final_weights != 0:
        print(f"Note: Weights for active components re-normalized from sum {current_sum_of_final_weights:.3f} to 1.0.")
        component_weights_series = component_weights_series / current_sum_of_final_weights
    elif current_sum_of_final_weights == 0 and not component_weights_series.empty:
        print("Error: Sum of final valid weights is zero. Cannot create meaningful index.")
        return pd.Series(dtype='float64', index=index_components_df.index)


    weighted_components_df = index_components_df.multiply(component_weights_series, axis='columns')
    simple_index_series = weighted_components_df.sum(axis=1)
    return simple_index_series

# --- Plotting Function (Saves to HTML) ---
def plot_step2_results(user_index_series, benchmark_normalized_series, 
                       user_index_name="Your Custom Portfolio", benchmark_name=DEFAULT_BENCHMARK_NAME,
                       filename_prefix="step2_plot"):
    if user_index_series.empty:
        print("Cannot plot: User index series is empty.")
        return

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=user_index_series.index, y=user_index_series,
                              mode='lines', name=user_index_name, line=dict(width=2)))
    if benchmark_normalized_series is not None and not benchmark_normalized_series.empty:
        fig1.add_trace(go.Scatter(x=benchmark_normalized_series.index, y=benchmark_normalized_series,
                                  mode='lines', name=benchmark_name + " (Normalized)",
                                  line=dict(dash='dash', color='grey')))
    fig1.update_layout(title=f"Step 2: {user_index_name} vs. {benchmark_name}",
                       xaxis_title="Date", yaxis_title="Normalized Value (Base 100)",
                       legend_title_text="Index")
    
    plot_filename_perf = f'{filename_prefix}_performance.html'
    try:
        plot(fig1, filename=plot_filename_perf, auto_open=False) 
        print(f"\nPerformance Plot saved to: {plot_filename_perf}")
    except Exception as e:
        print(f"Error saving performance plot: {e}")


    if benchmark_normalized_series is not None and not benchmark_normalized_series.empty:
        aligned_user_index, aligned_benchmark = user_index_series.align(benchmark_normalized_series, join='inner')
        if not aligned_user_index.empty and not aligned_benchmark.empty:
            safe_aligned_benchmark = aligned_benchmark.replace(0, np.nan)
            ratio_series = aligned_user_index / safe_aligned_benchmark
            ratio_series.dropna(inplace=True)
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
                plot_filename_ratio = f'{filename_prefix}_ratio.html'
                try:
                    plot(fig2, filename=plot_filename_ratio, auto_open=False)
                    print(f"Ratio Plot saved to: {plot_filename_ratio}")
                except Exception as e:
                    print(f"Error saving ratio plot: {e}")

            else: print("Could not calculate or plot the performance ratio (series empty after processing).")
        else: print("Could not align custom index and benchmark for ratio calculation (empty after align).")
    else: print("Benchmark data not available for ratio plot.")

# --- Main Execution Block for Step 2 ---
def run_foundational_portfolio_analysis(selected_stocks, base_weights_input, start_date_str_main, end_date_str_main): # Changed var names
    print(f"\n--- AlphaKnaut: Step 2 - Crafting Core Portfolio ({', '.join(selected_stocks)}) ---")
    if not selected_stocks: print("Error: Please provide a list of stocks."); return

    final_weights = {}
    if isinstance(base_weights_input, str) and base_weights_input.lower() == 'equal':
        if selected_stocks: final_weights = {ticker: 1.0 / len(selected_stocks) for ticker in selected_stocks}
        else: print("Error: Cannot use 'equal' weights with no selected stocks."); return
    elif isinstance(base_weights_input, dict):
        current_sum = sum(base_weights_input.values())
        if not np.isclose(current_sum, 1.0):
            print(f"Warning: Provided custom weights sum to {current_sum:.3f}, not 1.0.")
            if current_sum != 0:
                print("Normalizing weights to sum to 1.0...")
                final_weights = {ticker: weight / current_sum for ticker, weight in base_weights_input.items()}
            else:
                print("Error: Custom weights sum to 0. Using equal weights as fallback."); final_weights = {ticker: 1.0 / len(selected_stocks) for ticker in selected_stocks}
        else: final_weights = base_weights_input
    else: print("Error: Invalid 'base_weights_input'."); return
        
    print(f"Using Base Weights (ω): {final_weights}")

    # Ensure we only try to fetch tickers that will have weights
    tickers_with_weights = list(final_weights.keys())
    if not tickers_with_weights:
        print("Error: No tickers with weights to process.")
        return
        
    tickers_to_fetch = tickers_with_weights + [DEFAULT_BENCHMARK_TICKER]
    all_price_data = get_stock_data_step2(tickers_to_fetch, start_date_str_main, end_date_str_main)

    if all_price_data.empty: print("Failed to fetch price data. Aborting Step 2 analysis."); return
        
    # After fetching, ensure all tickers we based weights on are actually present
    final_valid_tickers_for_weights = [t for t in tickers_with_weights if t in all_price_data.columns]
    if len(final_valid_tickers_for_weights) != len(tickers_with_weights):
        print(f"Warning: Some initially weighted tickers were not found in downloaded data. Using: {final_valid_tickers_for_weights}")
        # Re-evaluate final_weights if some are missing
        if not final_valid_tickers_for_weights:
            print("Error: None of the weighted tickers have data. Aborting."); return
        
        # If using equal weights originally, re-calculate based on valid tickers
        if isinstance(base_weights_input, str) and base_weights_input.lower() == 'equal':
            final_weights = {ticker: 1.0 / len(final_valid_tickers_for_weights) for ticker in final_valid_tickers_for_weights}
        else: # For custom, filter and re-normalize
            filtered_custom_weights = {t:final_weights[t] for t in final_valid_tickers_for_weights}
            current_sum = sum(filtered_custom_weights.values())
            if current_sum != 0:
                final_weights = {t: w/current_sum for t,w in filtered_custom_weights.items()}
            else: # Should not happen if tickers are valid
                print("Error: Sum of filtered custom weights is zero. Aborting."); return
        print(f"Adjusted Base Weights (ω) after data validation: {final_weights}")


    if DEFAULT_BENCHMARK_TICKER not in all_price_data.columns:
        print(f"Error: Benchmark {DEFAULT_BENCHMARK_TICKER} data missing after download. Aborting.")
        return

    normalized_price_data = robust_normalize_prices_step2(all_price_data)
    if normalized_price_data.empty or normalized_price_data.isnull().all().all():
        print("Failed to normalize price data or result is all NaNs. Aborting Step 2 analysis."); return

    # Ensure we only use columns for which we have weights and which exist after normalization
    user_portfolio_components_normalized = normalized_price_data[[t for t in final_weights.keys() if t in normalized_price_data.columns]]
    benchmark_normalized = normalized_price_data.get(DEFAULT_BENCHMARK_TICKER)

    if user_portfolio_components_normalized.empty:
        print("No valid component data after normalization for selected weighted tickers. Aborting.")
        return

    user_simple_index = calculate_simple_weighted_index_step2(user_portfolio_components_normalized, final_weights)
    if user_simple_index.empty: print("Failed to calculate your custom simple index. Aborting."); return

    plot_step2_results(user_simple_index, benchmark_normalized, 
                       user_index_name="Your Core Portfolio (ω & P)", 
                       benchmark_name=DEFAULT_BENCHMARK_NAME,
                       filename_prefix=f"step2_{'_'.join(final_valid_tickers_for_weights)}_{start_date_str_main.replace('-','')}")
    print("\n--- Step 2 Analysis Complete ---")


if __name__ == "__main__":
    # Scenario 1: Equal Weights
    print("\n\n--- SCENARIO 1: EQUAL WEIGHTS ---")
    user_selected_stocks_eq = ['AAPL', 'MSFT', 'GOOGL', 'NVDA']
    user_weights_eq = 'equal'
    start_date_input_eq_str = "2023-01-01" # Use string directly
    end_date_input_eq_str = (pd.Timestamp.today() - pd.Timedelta(days=1)).strftime('%Y-%m-%d')

    run_foundational_portfolio_analysis(
        selected_stocks=user_selected_stocks_eq,
        base_weights_input=user_weights_eq,
        start_date_str_main=start_date_input_eq_str, # Pass strings
        end_date_str_main=end_date_input_eq_str
    )

    # Scenario 2: Custom Weights
    print("\n\n--- SCENARIO 2: CUSTOM WEIGHTS ---")
    user_selected_stocks_custom = ['AAPL', 'AMZN', 'TSLA']
    user_weights_custom = {'AAPL': 0.5, 'AMZN': 0.3, 'TSLA': 0.2}
    start_date_input_custom_str = "2023-06-01"
    
    run_foundational_portfolio_analysis(
        selected_stocks=user_selected_stocks_custom,
        base_weights_input=user_weights_custom,
        start_date_str_main=start_date_input_custom_str, # Pass strings
        end_date_str_main=end_date_input_eq_str # Reuse end date
    )

    # Scenario 3: Custom Weights that don't sum to 1 (to test normalization)
    print("\n\n--- SCENARIO 3: CUSTOM WEIGHTS (NON-NORMALIZED INPUT) ---")
    user_selected_stocks_custom_nn = ['GOOGL', 'META']
    # Test with a ticker that might have issues sometimes, e.g., less liquid or newer
    # user_selected_stocks_custom_nn = ['GOOGL', 'SNOW'] 
    user_weights_custom_nn = {'GOOGL': 0.7, 'META': 0.5} # Sums to 1.2
    
    run_foundational_portfolio_analysis(
        selected_stocks=user_selected_stocks_custom_nn,
        base_weights_input=user_weights_custom_nn,
        start_date_str_main=start_date_input_custom_str,
        end_date_str_main=end_date_input_eq_str
    )

    # Scenario 4: Single stock
    print("\n\n--- SCENARIO 4: SINGLE STOCK ---")
    user_selected_stocks_single = ['NVDA']
    user_weights_single = 'equal' # Will result in {'NVDA': 1.0}
    
    run_foundational_portfolio_analysis(
        selected_stocks=user_selected_stocks_single,
        base_weights_input=user_weights_single,
        start_date_str_main="2023-08-01",
        end_date_str_main=end_date_input_eq_str
    )
