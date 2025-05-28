# aki_dashboard.py

# --- Imports ---
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# from plotly.offline import init_notebook_mode, iplot # Removed for .py file, fig.show() will be used
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, VBox, HBox, Layout, HTML, FloatText, Button, DatePicker, SelectMultiple, RadioButtons, Checkbox
from IPython.display import display, clear_output # Requires an IPython environment (like Jupyter, VSCode with Jupyter ext, IPython console)
import numpy as np
import warnings

warnings.filterwarnings('ignore') # Suppress warnings, use with caution in production

# Note: For the interactive UI to work, this script should be run in an environment
# that supports ipywidgets and IPython.display, such as:
# - Jupyter Notebook or JupyterLab
# - VS Code with the Python and Jupyter extensions (run in Interactive Window or Notebook)
# - An IPython console
# If run with a standard Python interpreter, the UI parts will likely not render or will error.

# init_notebook_mode(connected=True) # Not needed if using fig.show() or if environment handles it

# --- Helper to create default series if index_ref is problematic ---
def create_default_series(index_ref, default_val=0.0, name=None):
    if index_ref is None or index_ref.empty:
        # Fallback to a generic short index if no valid reference is provided
        print(f"Warning: Invalid index_ref for creating default series for {name if name else 'Unnamed Series'}. Using generic index.")
        index_ref = pd.date_range(start='2020-01-01', periods=10, freq='B') # Business days
    return pd.Series(default_val, index=index_ref, name=name)

# --- GLOBAL DEFINITION for robust_normalize ---
def robust_normalize(data_series_or_df):
    if data_series_or_df is None or data_series_or_df.empty:
        # print(f"Robust Normalize: Input data is None or empty. Returning empty structure.")
        return pd.Series(dtype='float64') if isinstance(data_series_or_df, pd.Series) else pd.DataFrame()
    
    data_copy = data_series_or_df.copy() # Work on a copy
    
    # Attempt to get the first valid (non-NaN) row for normalization after forward fill
    first_valid_row_values = data_copy.ffill().iloc[0]
    
    if isinstance(data_copy, pd.DataFrame): # Input was a DataFrame
        if first_valid_row_values.empty: # Should not happen if data_copy wasn't empty
            print("Robust Normalize: DataFrame's first row (after ffill) is empty. Returning original DataFrame.")
            return data_copy 
        
        # For each column, normalize by its first valid non-zero value
        normalized_cols = {}
        for col in data_copy.columns:
            col_series = data_copy[col]
            first_val_col = col_series.ffill().iloc[0] # Get first non-NaN for this column
            
            if pd.isna(first_val_col) or first_val_col == 0:
                # Try to find first non-zero, non-NaN value in the column
                first_valid_nonzero_col = None
                for val in col_series.ffill().bfill(): # Iterate through filled values
                    if pd.notna(val) and val != 0:
                        first_valid_nonzero_col = val
                        break
                if first_valid_nonzero_col is not None:
                    # print(f"Robust Normalize (Col {col}): First valid value was 0/NaN, using {first_valid_nonzero_col} for normalization.")
                    normalized_cols[col] = (col_series / first_valid_nonzero_col) * 100
                else:
                    # print(f"Robust Normalize (Col {col}): Could not find valid non-zero value for normalization. Returning column * 100 or original.")
                    normalized_cols[col] = col_series * 100 # Or col_series.copy() if no scaling desired
            else: # First valid value is non-zero and not NaN
                normalized_cols[col] = (col_series / first_val_col) * 100
        return pd.DataFrame(normalized_cols)

    elif isinstance(data_copy, pd.Series): # Input was a Series
        first_val_series = first_valid_row_values # This is a scalar here
        
        if pd.isna(first_val_series) or first_val_series == 0:
            first_valid_nonzero_series = None
            for val in data_copy.ffill().bfill():
                if pd.notna(val) and val != 0:
                    first_valid_nonzero_series = val
                    break
            if first_valid_nonzero_series is not None:
                # print(f"Robust Normalize (Series {data_copy.name if hasattr(data_copy, 'name') else ''}): First valid was 0/NaN, using {first_valid_nonzero_series}.")
                return (data_copy / first_valid_nonzero_series) * 100
            else:
                # print(f"Robust Normalize (Series {data_copy.name if hasattr(data_copy, 'name') else ''}): No valid non-zero value. Returning series * 100 or original.")
                return data_copy * 100
        else:
            return (data_copy / first_val_series) * 100
    else: # Should not happen
        return data_series_or_df 

# --- Enhanced data fetching ---
def get_enhanced_market_data(tickers_list_input, start_date_str, end_date_str):
    tickers_list = list(tickers_list_input) 
    if not tickers_list:
        print("No tickers selected for download.")
        return pd.DataFrame(), pd.DataFrame()
    
    market_indicators_map = {'^VIX': 'VIX', '^TNX': 'TNX', 'DX-Y.NYB': 'DXY'}
    all_assets_to_download = tickers_list + list(market_indicators_map.keys())
    
    print(f"Downloading data for: {', '.join(all_assets_to_download)} from {start_date_str} to {end_date_str}...")
    
    try:
        raw_data = yf.download(all_assets_to_download, start=start_date_str, end=end_date_str, auto_adjust=True, progress=False, timeout=30)
        
        if raw_data.empty:
            print("No data downloaded from yfinance for any ticker.")
            return pd.DataFrame(), pd.DataFrame()

        price_data_cols = {}
        indicator_data_cols = {}
        is_multi_index = isinstance(raw_data.columns, pd.MultiIndex)
        
        for asset in all_assets_to_download:
            asset_data = None
            if is_multi_index:
                potential_col_names = [('Close', asset), ('Adj Close', asset)] 
                for col_name in potential_col_names:
                    if col_name in raw_data.columns:
                        asset_data = raw_data[col_name]
                        break
                if asset_data is None and asset in raw_data.columns.get_level_values(1): 
                     for level0_name in raw_data.columns.levels[0]:
                         if (level0_name, asset) in raw_data.columns:
                             asset_data = raw_data[(level0_name, asset)]
                             break
            elif asset in raw_data.columns: 
                asset_data = raw_data[asset]

            if asset_data is not None and not asset_data.isnull().all():
                if asset in tickers_list:
                    price_data_cols[asset] = asset_data
                elif asset in market_indicators_map:
                    indicator_data_cols[market_indicators_map[asset]] = asset_data
            
        price_df = pd.DataFrame(price_data_cols)
        market_df = pd.DataFrame(indicator_data_cols)
        
        base_index_for_synthetic = price_df.index if not price_df.empty else pd.date_range(start=start_date_str, end=end_date_str, freq='B')
        if base_index_for_synthetic.empty: # If still empty (e.g., start=end date)
            base_index_for_synthetic = pd.date_range(start=start_date_str, periods=max(1, (pd.to_datetime(end_date_str) - pd.to_datetime(start_date_str)).days + 1), freq='B')


        for indicator_yf, indicator_name in market_indicators_map.items():
            if indicator_name not in market_df.columns or market_df[indicator_name].isnull().all():
                print(f"Warning: {indicator_name} data missing or empty, using synthetic {indicator_name}.")
                synthetic_series = None
                if indicator_name == 'VIX':   synthetic_series = pd.Series(np.random.normal(20, 5, len(base_index_for_synthetic)), index=base_index_for_synthetic).clip(10, 50)
                elif indicator_name == 'TNX': synthetic_series = pd.Series(np.random.normal(2.5, 0.5, len(base_index_for_synthetic)), index=base_index_for_synthetic).clip(0.5, 5)
                elif indicator_name == 'DXY': synthetic_series = pd.Series(np.random.normal(100, 3, len(base_index_for_synthetic)), index=base_index_for_synthetic).clip(90, 110)
                if synthetic_series is not None: market_df[indicator_name] = synthetic_series
        
        if not price_df.empty: price_df = price_df.ffill().bfill().dropna(how='all')
        if not market_df.empty: market_df = market_df.ffill().bfill().dropna(how='all')
        
        if not price_df.empty and not market_df.empty:
            common_index = price_df.index.intersection(market_df.index)
            if not common_index.empty:
                price_df = price_df.loc[common_index]
                market_df = market_df.loc[common_index]
            else:
                print("Warning: No common index. Reindexing market_df to price_df's index.")
                market_df = market_df.reindex(price_df.index).ffill().bfill()
        elif not price_df.empty and market_df.empty:
             print("Warning: Market indicators DataFrame became empty. Creating with NaNs based on price_df index.")
             market_df = pd.DataFrame(index=price_df.index, columns=list(market_indicators_map.values())) # Will be filled by synthetic later if still empty
        
        return price_df, market_df
        
    except Exception as e:
        print(f"CRITICAL Error during enhanced download: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame(), pd.DataFrame()

# --- Dynamic Component Calculation Functions ---
def calculate_dynamic_psi(vix_data_series, index_ref, lookback_days=60, eta=0.3):
    if vix_data_series is None or vix_data_series.empty:
        return create_default_series(index_ref, 1.0, name="Psi_Default")
    vix_aligned = vix_data_series.reindex(index_ref).ffill().bfill()
    if vix_aligned.isnull().all():
        return create_default_series(index_ref, 1.0, name="Psi_Default_NaN_Align")
    vix_rolling_avg = vix_aligned.rolling(window=lookback_days, min_periods=max(1, lookback_days//3)).mean()
    vix_rolling_avg = vix_rolling_avg.replace(0, np.nan).ffill().bfill() 
    if vix_rolling_avg.isnull().all(): 
        return create_default_series(index_ref, 1.0, name="Psi_Default_NaN_RollAvg")
    vix_normalized = (vix_aligned / vix_rolling_avg) - 1
    psi_dynamic = 1 + np.tanh(vix_normalized.fillna(0)) * eta
    return psi_dynamic.reindex(index_ref).fillna(1.0)

def calculate_adaptive_weights(price_data_df, base_weights_dict, sentiment_strength_param=0.1, momentum_lookback=20):
    index_ref = price_data_df.index if not price_data_df.empty else None
    if price_data_df.empty or not base_weights_dict:
        return {ticker: create_default_series(index_ref, base_weights_dict.get(ticker, 0), name=f"Weight_{ticker}") for ticker in base_weights_dict} if base_weights_dict else {}
    adaptive_weights_output = {}
    returns_df = price_data_df.pct_change(momentum_lookback).fillna(0) 
    for ticker in price_data_df.columns:
        if ticker in base_weights_dict:
            momentum_score_series = returns_df[ticker].rolling(window=5, min_periods=1).mean().fillna(0)
            sentiment_noise_series = pd.Series(np.random.normal(0, 0.05, len(momentum_score_series)), index=momentum_score_series.index)
            weight_multiplier_series = (1 + (momentum_score_series * 0.5) + (sentiment_noise_series * sentiment_strength_param)).clip(0.5, 1.5)
            weight_multiplier_aligned = weight_multiplier_series.reindex(index_ref).fillna(1.0)
            adaptive_weights_output[ticker] = base_weights_dict[ticker] * weight_multiplier_aligned
    return adaptive_weights_output

def calculate_dynamic_alpha(price_data_df, options_proxy_strength_param=0.05, momentum_lookback=10):
    index_ref = price_data_df.index if not price_data_df.empty else None
    if price_data_df.empty: return {ticker: create_default_series(index_ref, 1.0, name=f"Alpha_{ticker}") for ticker in price_data_df.columns} if not price_data_df.empty else {} # Return empty if no columns
    dynamic_alphas_output = {}
    for ticker in price_data_df.columns:
        returns_series = price_data_df[ticker].pct_change().fillna(0)
        momentum_series = returns_series.rolling(window=momentum_lookback, min_periods=1).mean().fillna(0)
        options_flow_series = pd.Series(np.random.normal(0, 0.02, len(price_data_df)), index=price_data_df.index)
        alpha_dynamic_series = 1 + (momentum_series * options_flow_series * options_proxy_strength_param * 10)
        alpha_dynamic_series = alpha_dynamic_series.clip(0.75, 1.5) 
        dynamic_alphas_output[ticker] = alpha_dynamic_series.reindex(index_ref).fillna(1.0)
    return dynamic_alphas_output

def calculate_dynamic_beta(benchmark_data_series, component_data_df, index_ref, lookback_window=60):
    if benchmark_data_series is None or benchmark_data_series.empty or \
       component_data_df is None or component_data_df.empty or \
       index_ref is None or index_ref.empty:
        return create_default_series(index_ref, 1.0, name="Beta_Default_Initial_Empty")
    benchmark_aligned = benchmark_data_series.reindex(index_ref).ffill().bfill()
    component_aligned = component_data_df.reindex(index_ref).ffill().bfill()
    is_benchmark_problematic = benchmark_aligned.isnull().all()
    is_component_problematic = component_aligned.empty or (isinstance(component_aligned, pd.DataFrame) and component_aligned.isnull().all().all())
    if is_benchmark_problematic or is_component_problematic:
        return create_default_series(index_ref, 1.0, name="Beta_Default_NaN_Align")
    benchmark_returns = benchmark_aligned.pct_change().fillna(0)
    component_avg_returns = component_aligned.pct_change().mean(axis=1).fillna(0)
    if (benchmark_returns == 0).all():
        return create_default_series(index_ref, 1.0, name="Beta_Default_Zero_Returns")
    min_periods_beta = max(5, lookback_window // 3) 
    rolling_cov = benchmark_returns.rolling(window=lookback_window, min_periods=min_periods_beta).cov(component_avg_returns)
    rolling_var = benchmark_returns.rolling(window=lookback_window, min_periods=min_periods_beta).var()
    dynamic_beta_series = create_default_series(index_ref, 1.0, name="Beta_Initial_Default") 
    valid_var_mask = (~rolling_var.isnull()) & (rolling_var != 0)
    if valid_var_mask.any(): 
        dynamic_beta_series.loc[valid_var_mask] = rolling_cov.loc[valid_var_mask] / rolling_var.loc[valid_var_mask]
    dynamic_beta_series = dynamic_beta_series.fillna(method='ffill').fillna(method='bfill').fillna(1.0)
    return dynamic_beta_series.reindex(index_ref).fillna(1.0)

def calculate_cross_asset_theta(market_indicators_df, index_ref, bond_w=0.1, currency_w=0.1):
    if market_indicators_df is None or market_indicators_df.empty:
        return create_default_series(index_ref, 0.0, name="Theta_Default")
    theta_components_series = create_default_series(index_ref, 0.0)
    if 'TNX' in market_indicators_df.columns and not market_indicators_df['TNX'].isnull().all():
        tnx_aligned = market_indicators_df['TNX'].reindex(index_ref).ffill().bfill()
        bond_momentum = tnx_aligned.pct_change(10).fillna(0)
        theta_components_series = theta_components_series.add(bond_w * bond_momentum, fill_value=0)
    if 'DXY' in market_indicators_df.columns and not market_indicators_df['DXY'].isnull().all():
        dxy_aligned = market_indicators_df['DXY'].reindex(index_ref).ffill().bfill()
        dxy_momentum = dxy_aligned.pct_change(5).fillna(0)
        theta_components_series = theta_components_series.add(currency_w * dxy_momentum, fill_value=0)
    return theta_components_series.reindex(index_ref).fillna(0.0)

def calculate_behavioral_phi(price_data_df, market_indicators_df, index_ref):
    if price_data_df.empty:
        return create_default_series(index_ref, 0.0, name="Phi_Default")
    phi_total_series = create_default_series(index_ref, 0.0)
    if market_indicators_df is not None and 'VIX' in market_indicators_df.columns and not market_indicators_df['VIX'].isnull().all():
        vix_aligned = market_indicators_df['VIX'].reindex(index_ref).ffill().bfill()
        vix_momentum = vix_aligned.pct_change(3).fillna(0)
        phi_total_series = phi_total_series.add(-vix_momentum * 0.05, fill_value=0)
    volume_proxy_series = price_data_df.pct_change().abs().mean(axis=1).fillna(0)
    insider_signal_series = (volume_proxy_series - volume_proxy_series.rolling(20, min_periods=1).mean().fillna(0)) * 0.02
    phi_total_series = phi_total_series.add(insider_signal_series.reindex(index_ref).fillna(0), fill_value=0)
    dark_pool_signal_series = pd.Series(np.random.normal(0, 0.005, len(index_ref)), index=index_ref)
    phi_total_series = phi_total_series.add(dark_pool_signal_series, fill_value=0)
    return phi_total_series.reindex(index_ref).fillna(0.0)

def calculate_ai_forecast_omega(price_data_df, market_indicators_df, index_ref, forecast_str_param=0.05):
    if price_data_df.empty:
        return create_default_series(index_ref, 0.0, name="Omega_Default")
    omega_components_series = create_default_series(index_ref, 0.0)
    if market_indicators_df is not None and not market_indicators_df.empty:
        mi_aligned = market_indicators_df.reindex(index_ref).ffill().bfill()
        if not mi_aligned.empty and not mi_aligned.isnull().all().all(): 
            economic_composite = mi_aligned.mean(axis=1, skipna=True).pct_change(5).fillna(0)
            omega_components_series = omega_components_series.add(economic_composite * 0.1, fill_value=0)
    satellite_signal_series = pd.Series(np.random.normal(0, 0.002, len(index_ref)), index=index_ref)
    omega_components_series = omega_components_series.add(satellite_signal_series * 0.1, fill_value=0)
    price_momentum_series = price_data_df.pct_change(10).mean(axis=1).fillna(0)
    sentiment_proxy_series = (price_momentum_series - price_momentum_series.rolling(30, min_periods=1).mean().fillna(0))
    omega_components_series = omega_components_series.add(sentiment_proxy_series.reindex(index_ref).fillna(0) * 0.1, fill_value=0)
    innovation_signal_series = pd.Series(np.random.normal(0.0005, 0.001, len(index_ref)), index=index_ref)
    omega_components_series = omega_components_series.add(innovation_signal_series * 0.1, fill_value=0)
    return (omega_components_series * forecast_str_param).reindex(index_ref).fillna(0.0)

# --- Main AKI Calculation Function ---
def create_graduate_level_aki(price_data_df, benchmark_data_series, market_indicators_df, base_weights_dict, 
                              psi_eta_ui_val, momentum_strength_ui_val, forecast_strength_ui_val):
    if price_data_df.empty:
        return pd.Series(dtype='float64'), "AKI (No Component Data)", {}
    
    common_index = price_data_df.index
    if common_index.empty:
        print("Error: Price data index is empty for AKI. Cannot proceed.")
        return pd.Series(dtype='float64'), "AKI (Empty Index)", {}

    print("AKI: Normalizing data...")
    normalized_components_df = robust_normalize(price_data_df)
    normalized_benchmark_series = robust_normalize(benchmark_data_series)
    normalized_components_df = normalized_components_df.reindex(common_index).ffill().bfill()
    normalized_benchmark_series = normalized_benchmark_series.reindex(common_index).ffill().bfill()

    if normalized_components_df.empty : 
        print("Error: Normalized components are empty after robust_normalize and reindex."); return pd.Series(dtype='float64'), "AKI (Norm Error Comp)", {}
    if normalized_benchmark_series.empty and benchmark_data_series is not None and not benchmark_data_series.empty:
        print("Warning: Normalized benchmark is empty, using zeros.")
        normalized_benchmark_series = create_default_series(common_index, 0.0, name="Norm_Bench_Default")

    print("AKI: Calculating dynamic components...")
    vix_series_input = market_indicators_df.get('VIX', create_default_series(common_index, 20, name="VIX_Default_Input"))
    psi_series = calculate_dynamic_psi(vix_series_input, common_index, eta=psi_eta_ui_val)
    adaptive_weights_map = calculate_adaptive_weights(price_data_df, base_weights_dict, sentiment_strength_param=momentum_strength_ui_val)
    dynamic_alphas_map = calculate_dynamic_alpha(price_data_df, options_proxy_strength_param=momentum_strength_ui_val)
    beta_series = calculate_dynamic_beta(benchmark_data_series, price_data_df, common_index) # price_data_df is component_data
    theta_series = calculate_cross_asset_theta(market_indicators_df, common_index)
    phi_series = calculate_behavioral_phi(price_data_df, market_indicators_df, common_index)
    omega_series = calculate_ai_forecast_omega(price_data_df, market_indicators_df, common_index, forecast_str_param=forecast_strength_ui_val)
    
    all_series_for_aki = {'psi': psi_series, 'beta': beta_series, 'theta': theta_series, 'phi': phi_series, 'omega': omega_series}
    for name, s_val in all_series_for_aki.items():
        if s_val is None or s_val.empty:
            all_series_for_aki[name] = create_default_series(common_index, 0.0 if name not in ['psi', 'beta'] else 1.0, name=f"{name}_Default_Reindex")
        else:
            all_series_for_aki[name] = s_val.reindex(common_index).ffill().bfill().fillna(0.0 if name not in ['psi', 'beta'] else 1.0)

    psi_s, beta_s, theta_s, phi_s, omega_s = (all_series_for_aki['psi'], all_series_for_aki['beta'], 
                                             all_series_for_aki['theta'], all_series_for_aki['phi'], 
                                             all_series_for_aki['omega'])

    asset_performance_sum = create_default_series(common_index, 0.0, name="AssetPerfSum_Default")
    for ticker in price_data_df.columns:
        if ticker in base_weights_dict:
            weight_s = adaptive_weights_map.get(ticker, pd.Series(base_weights_dict[ticker], index=common_index))
            weight_s = weight_s.reindex(common_index).ffill().bfill().fillna(base_weights_dict[ticker])
            alpha_s = dynamic_alphas_map.get(ticker, pd.Series(1.0, index=common_index))
            alpha_s = alpha_s.reindex(common_index).ffill().bfill().fillna(1.0)
            
            norm_comp_ticker = normalized_components_df.get(ticker)
            if norm_comp_ticker is None or norm_comp_ticker.empty or norm_comp_ticker.isnull().all(): continue
            performance_ratio_s = norm_comp_ticker / 100.0
            
            powered_performance_s = performance_ratio_s.pow(alpha_s)
            weighted_performance_s = weight_s * powered_performance_s
            asset_performance_sum = asset_performance_sum.add(weighted_performance_s.fillna(0), fill_value=0)
    
    benchmark_ratio_s = (normalized_benchmark_series / 100.0) if not normalized_benchmark_series.empty else create_default_series(common_index, 0.0, name="BenchRatio_Default")
    benchmark_term = beta_s * benchmark_ratio_s
    
    print("AKI: Assembling final series...")
    aki_series_final = (psi_s * asset_performance_sum) - benchmark_term + theta_s + phi_s + omega_s
    
    components_for_plot_final = {
        'psi': psi_s, 'beta': beta_s, 'theta': theta_s, 
        'phi': phi_s, 'omega': omega_s,
        'adaptive_weights': adaptive_weights_map, 
        'dynamic_alphas': dynamic_alphas_map    
    }
    return aki_series_final.dropna(), "Graduate-Level Dynamic AKI", components_for_plot_final

# --- UI Components and Dashboard Logic ---
TECH_TICKERS_JUPYTER = {
    'AAPL': 'Apple Inc.', 'MSFT': 'Microsoft Corp.', 'GOOGL': 'Alphabet Inc.', 'AMZN': 'Amazon.com Inc.', 
    'NVDA': 'NVIDIA Corporation', 'META': 'Meta Platforms Inc.', 'TSLA': 'Tesla Inc.'}

start_date_widget = DatePicker(description='Start Date:', value=pd.to_datetime("2023-01-01")) # Shorter default
end_date_widget = DatePicker(description='End Date:', value=pd.Timestamp.today() - pd.Timedelta(days=1))
ticker_options = [(f"{name} ({ticker})", ticker) for ticker, name in TECH_TICKERS_JUPYTER.items()]
selected_tickers_widget = SelectMultiple(options=ticker_options, value=['AAPL', 'MSFT', 'NVDA'], description='Select Stocks:', rows=min(7, len(TECH_TICKERS_JUPYTER)))
enable_advanced_widget = Checkbox(description='Tune Dynamic Parameters:', value=True, indent=False) # Default to true to see params
psi_eta_widget = FloatText(value=0.3, description='Ψ Sensitivity (η):', step=0.05, layout=Layout(width='350px')) # Wider
momentum_strength_widget = FloatText(value=0.1, description='ω/α Mom. Factor:', step=0.02, layout=Layout(width='350px'))
forecast_strength_widget = FloatText(value=0.05, description='Ω AI Factor:', step=0.01, layout=Layout(width='350px'))
advanced_controls_vbox = VBox([psi_eta_widget, momentum_strength_widget, forecast_strength_widget])
run_button = Button(description="Run AKI Analysis", button_style='success', icon='play') 
output_area = widgets.Output()

def toggle_advanced_controls(change): # change can be None on initial call
    display_style = 'flex' if enable_advanced_widget.value else 'none'
    for w_adv in advanced_controls_vbox.children: w_adv.layout.display = display_style

def on_run_button_clicked(b): 
    # Get current values from widgets when button is clicked
    update_analysis_main(
        start_date_widget.value, end_date_widget.value, selected_tickers_widget.value,
        enable_advanced_widget.value, psi_eta_widget.value, 
        momentum_strength_widget.value, forecast_strength_widget.value
    )

def update_analysis_main(start_date, end_date, selected_tickers, 
                         advanced_enabled, psi_eta_val, momentum_strength_val, forecast_strength_val):
    with output_area:
        clear_output(wait=True) # Requires IPython.display
        if not selected_tickers: print("Please select at least one stock."); return
        
        print(f"🚀 Initializing AKI Analysis for {', '.join(selected_tickers)} from {start_date.date()} to {end_date.date()}...")
        print(f"Parameters: η={psi_eta_val if advanced_enabled else 'Default'}, MomStr={momentum_strength_val if advanced_enabled else 'Default'}, ForecastStr={forecast_strength_val if advanced_enabled else 'Default'}")

        
        all_fetch_tickers = list(selected_tickers) + ['^GSPC']
        price_data_df, market_indicators_df = get_enhanced_market_data(
            all_fetch_tickers, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
        )
        
        if price_data_df.empty: print("❌ No price data. Analysis aborted."); return
        
        # Ensure selected_tickers are columns in price_data_df before trying to select
        valid_selected_tickers = [t for t in selected_tickers if t in price_data_df.columns]
        if not valid_selected_tickers:
            print(f"❌ None of the selected stocks ({', '.join(selected_tickers)}) found in downloaded price data. Available: {', '.join(price_data_df.columns)}. Analysis aborted.")
            return
        component_data_df = price_data_df[valid_selected_tickers]
        benchmark_data_series = price_data_df.get('^GSPC', pd.Series(dtype='float64'))

        if component_data_df.empty: print("❌ No component data after filtering. Analysis aborted."); return

        # Base weights for selected valid tickers
        base_weights_dict = {ticker: 1.0/len(valid_selected_tickers) for ticker in valid_selected_tickers}
            
        # Use default values if advanced tuning is not enabled
        current_psi_eta = psi_eta_val if advanced_enabled else 0.3 # Default eta
        current_mom_strength = momentum_strength_val if advanced_enabled else 0.1 # Default
        current_forecast_strength = forecast_strength_val if advanced_enabled else 0.05 # Default

        aki_series, title, components_dict = create_graduate_level_aki(
            component_data_df, benchmark_data_series, market_indicators_df, base_weights_dict,
            psi_eta_ui_val=current_psi_eta, 
            momentum_strength_ui_val=current_mom_strength, 
            forecast_strength_ui_val=current_forecast_strength
        )
            
        if aki_series.empty: print("❌ AKI calculation resulted in empty series."); return

        print("📊 Plotting results...")
        fig = make_subplots(
            rows=3, cols=2, column_widths=[0.5, 0.5], row_heights=[0.4, 0.3, 0.3],
            subplot_titles=['AKI vs S&P 500', 'Ψ Market Regime', 'β Dynamic Beta', 
                            'Θ Cross-Asset', 'Ω AI Forecast', 'Φ Behavioral'],
            specs=[[{"colspan": 2, "secondary_y":True}, None], [{}, {}], [{}, {}]]
        )
        fig.add_trace(go.Scatter(x=aki_series.index, y=aki_series, name='AKI', line=dict(width=2, color='blue')), 
                      secondary_y=False, row=1, col=1)
        
        if benchmark_data_series is not None and not benchmark_data_series.empty:
            norm_bench_series = robust_normalize(benchmark_data_series) 
            if not norm_bench_series.empty:
                fig.add_trace(go.Scatter(x=norm_bench_series.index, y=norm_bench_series, name='S&P 500 (Norm)', line=dict(dash='dash', color='rgba(128,128,128,0.7)')),
                              secondary_y=True, row=1, col=1)
        
        plot_details = [
            ('psi', 'Ψ Market Regime', 2, 1), ('beta', 'β Dynamic Beta', 2, 2),
            ('theta', 'Θ Cross-Asset', 3, 1), ('omega', 'Ω AI Forecast', 3, 2)
            # ('phi', 'Φ Behavioral', R, C) # To add Phi, might need 4th row or different layout
        ]
        # Plot Phi (behavioral) separately or add to layout if space permits
        phi_data_series = components_dict.get('phi')
        if phi_data_series is not None and not phi_data_series.empty:
            # For simplicity, adding it to the Omega plot's subplot (3,2) or create a new row
            # Current layout has subplot_titles defined for 3x2.
            # Let's add it to the 3,2 subplot for now, making that subplot have two traces.
            # Ideally, the subplot_titles and layout should be adjusted if Phi gets its own dedicated subplot.
            # As it is, this will overlay Phi on the "Omega AI Forecast" subplot.
            # A better approach would be to make it 4 rows, or 3 rows and 3 columns if space allows.
            # For this conversion, I'll plot it on 3,2.
            # The original subplot title was 'Ω AI Forecast', 'Φ Behavioral'.
            # The plot title for (3,2) is currently 'Ω AI Forecast'.
            # Let's add it to a new spot, which means changing the subplot structure or adding it to an existing one.
            # The original notebook has a subplot title 'Φ Behavioral' suggesting it was intended to be plotted.
            # I will assume it can be added to the 3,2 plot.
             fig.add_trace(go.Scatter(x=phi_data_series.index, y=phi_data_series, name='Φ Behavioral'), row=3, col=2)


        for comp_key, comp_name_plot, r, c in plot_details:
            comp_data_series = components_dict.get(comp_key)
            if comp_data_series is not None and not comp_data_series.empty:
                 fig.add_trace(go.Scatter(x=comp_data_series.index, y=comp_data_series, name=comp_name_plot), row=r, col=c)
            
        fig.update_layout(height=1000, title_text="🎓 AlphaKnaut Dynamic AKI Analysis", showlegend=True,
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                          yaxis_title='AKI Value / Component Value', yaxis2_title='S&P 500 Norm.')
        fig.update_yaxes(secondary_y=True, showgrid=False, row=1, col=1, title_font=dict(size=10), tickfont=dict(size=9))
        fig.update_yaxes(secondary_y=False, showgrid=True, row=1, col=1, title_font=dict(size=10), tickfont=dict(size=9))
        for r_idx in [2,3]:
            for c_idx in [1,2]:
                 fig.update_yaxes(title_text="Factor Value", row=r_idx, col=c_idx, title_font=dict(size=10), tickfont=dict(size=9))

        fig.show() # Replaces iplot(fig) for .py files, typically opens in browser or IDE viewer
            
        if not aki_series.empty:
            print(f"\n📈 AKI Summary ({start_date.date()} to {end_date.date()}):")
            print(f"   Start AKI: {aki_series.iloc[0]:.3f}, End AKI: {aki_series.iloc[-1]:.3f}")
            print(f"   AKI Min: {aki_series.min():.3f}, AKI Max: {aki_series.max():.3f}, AKI Std Dev: {aki_series.std():.3f}")

# --- Main execution block for UI ---
if __name__ == "__main__":
    # This part sets up the UI. It requires an IPython-compatible environment.
    try:
        get_ipython() # Check if running in an IPython environment
        
        enable_advanced_widget.observe(toggle_advanced_controls, names='value')
        run_button.on_click(on_run_button_clicked)

        controls_vbox = VBox([
            HTML("<h1>🎓 AlphaKnaut Index (AKI): Graduate Edition</h1>"), HBox([start_date_widget, end_date_widget]),
            selected_tickers_widget, HTML("<h4>Tune Dynamic Component Parameters:</h4>"), enable_advanced_widget,
            advanced_controls_vbox, run_button, HTML("<hr>")])

        display(controls_vbox) # Requires IPython.display
        display(output_area)   # Requires IPython.display

        toggle_advanced_controls(None) # Set initial visibility of advanced controls
        print("AKI Dashboard Initialized. Adjust parameters and click 'Run AKI Analysis' to start.")
        print("Note: This script's interactive UI works best in Jupyter, VSCode (Jupyter extension), or an IPython console.")

    except NameError:
        print("This script contains an interactive UI using ipywidgets.")
        print("To use the UI, please run this script in an environment that supports ipywidgets,")
        print("such as Jupyter Notebook, JupyterLab, VS Code with the Python and Jupyter extensions, or an IPython console.")
        print("\nThe core calculation functions can still be imported and used programmatically if this script is treated as a module.")
        print("\nExample (if you were to import this script as 'aki_module'):")
        print("  from aki_dashboard import get_enhanced_market_data, create_graduate_level_aki")
        print("  # ... then call functions with appropriate data ...")