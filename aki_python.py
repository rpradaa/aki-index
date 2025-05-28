import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import warnings
from datetime import date, timedelta

warnings.filterwarnings('ignore') # Suppress warnings, use with caution in production

# --- Helper to create default series if index_ref is problematic ---
def create_default_series(index_ref, default_val=0.0, name=None):
    if index_ref is None or index_ref.empty:
        st.warning(f"Warning: Invalid index_ref for creating default series for {name if name else 'Unnamed Series'}. Using generic index.")
        index_ref = pd.date_range(start='2020-01-01', periods=10, freq='B') # Business days
    return pd.Series(default_val, index=index_ref, name=name)

# --- GLOBAL DEFINITION for robust_normalize ---
def robust_normalize(data_series_or_df):
    if data_series_or_df is None or data_series_or_df.empty:
        return pd.Series(dtype='float64') if isinstance(data_series_or_df, pd.Series) else pd.DataFrame()
    
    data_copy = data_series_or_df.copy() 
    first_valid_row_values = data_copy.ffill().iloc[0] if not data_copy.empty else (pd.Series() if isinstance(data_copy, pd.Series) else pd.DataFrame()) # Handle empty after copy
    
    if isinstance(data_copy, pd.DataFrame):
        if data_copy.empty or first_valid_row_values.empty: 
            st.warning("Robust Normalize: DataFrame or its first row (after ffill) is empty. Returning original DataFrame.")
            return data_copy 
        
        normalized_cols = {}
        for col in data_copy.columns:
            col_series = data_copy[col]
            if col_series.empty:
                normalized_cols[col] = pd.Series(dtype='float64', index=data_copy.index) # empty series for empty column
                continue

            first_val_col = col_series.ffill().iloc[0] if not col_series.ffill().empty else np.nan
            
            if pd.isna(first_val_col) or first_val_col == 0:
                first_valid_nonzero_col = None
                # Check if ffill().bfill() is not empty before iterating
                filled_bfilled_series = col_series.ffill().bfill()
                if not filled_bfilled_series.empty:
                    for val in filled_bfilled_series: 
                        if pd.notna(val) and val != 0:
                            first_valid_nonzero_col = val
                            break
                if first_valid_nonzero_col is not None:
                    normalized_cols[col] = (col_series / first_valid_nonzero_col) * 100
                else:
                    normalized_cols[col] = col_series * 100 
            else:
                normalized_cols[col] = (col_series / first_val_col) * 100
        return pd.DataFrame(normalized_cols)

    elif isinstance(data_copy, pd.Series):
        if data_copy.empty:
            st.warning(f"Robust Normalize: Series {(data_copy.name if hasattr(data_copy, 'name') else '')} is empty. Returning empty series.")
            return pd.Series(dtype='float64')

        first_val_series = first_valid_row_values # This is a scalar here
        
        if pd.isna(first_val_series) or first_val_series == 0:
            first_valid_nonzero_series = None
            filled_bfilled_series = data_copy.ffill().bfill()
            if not filled_bfilled_series.empty:
                for val in filled_bfilled_series:
                    if pd.notna(val) and val != 0:
                        first_valid_nonzero_series = val
                        break
            if first_valid_nonzero_series is not None:
                return (data_copy / first_valid_nonzero_series) * 100
            else:
                return data_copy * 100
        else:
            return (data_copy / first_val_series) * 100
    else: 
        return data_series_or_df

# --- Enhanced data fetching ---
@st.cache_data(ttl=3600) # Cache data for 1 hour
def get_enhanced_market_data(tickers_list_input, start_date_str, end_date_str):
    tickers_list = list(tickers_list_input) 
    if not tickers_list:
        st.warning("No tickers selected for download.")
        return pd.DataFrame(), pd.DataFrame()
    
    market_indicators_map = {'^VIX': 'VIX', '^TNX': 'TNX', 'DX-Y.NYB': 'DXY'}
    all_assets_to_download = tickers_list + list(market_indicators_map.keys())
    
    st.info(f"Downloading data for: {', '.join(all_assets_to_download)} from {start_date_str} to {end_date_str}...")
    
    try:
        raw_data = yf.download(all_assets_to_download, start=start_date_str, end=end_date_str, auto_adjust=True, progress=False, timeout=30)
        
        if raw_data.empty:
            st.error("No data downloaded from yfinance for any ticker.")
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
            # Handle case where yf.download returns columns without a level for single tickers when multiple are requested
            elif not is_multi_index and asset in raw_data.columns:
                 asset_data = raw_data[asset]
            # If yf.download returns a simple DataFrame for single asset, but we expect 'Close'
            elif isinstance(raw_data, pd.DataFrame) and 'Close' in raw_data.columns and len(all_assets_to_download) == 1 and all_assets_to_download[0] == asset:
                asset_data = raw_data['Close']


            if asset_data is not None and not asset_data.isnull().all():
                if asset in tickers_list:
                    price_data_cols[asset] = asset_data
                elif asset in market_indicators_map:
                    indicator_data_cols[market_indicators_map[asset]] = asset_data
            
        price_df = pd.DataFrame(price_data_cols)
        market_df = pd.DataFrame(indicator_data_cols)
        
        base_index_for_synthetic = price_df.index if not price_df.empty else pd.date_range(start=start_date_str, end=end_date_str, freq='B')
        if base_index_for_synthetic.empty: 
            base_index_for_synthetic = pd.date_range(start=start_date_str, periods=max(1, (pd.to_datetime(end_date_str) - pd.to_datetime(start_date_str)).days + 1), freq='B')


        for indicator_yf, indicator_name in market_indicators_map.items():
            if indicator_name not in market_df.columns or market_df[indicator_name].isnull().all():
                st.warning(f"Warning: {indicator_name} data missing or empty, using synthetic {indicator_name}.")
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
                st.warning("Warning: No common index. Reindexing market_df to price_df's index.")
                market_df = market_df.reindex(price_df.index).ffill().bfill()
        elif not price_df.empty and market_df.empty:
             st.warning("Warning: Market indicators DataFrame became empty. Creating with NaNs based on price_df index.")
             market_df = pd.DataFrame(index=price_df.index, columns=list(market_indicators_map.values()))
        
        return price_df, market_df
        
    except Exception as e:
        st.error(f"CRITICAL Error during enhanced download: {e}")
        import traceback
        st.text(traceback.format_exc())
        return pd.DataFrame(), pd.DataFrame()

# --- Dynamic Component Calculation Functions (largely unchanged, added st.warning for defaults) ---
def calculate_dynamic_psi(vix_data_series, index_ref, lookback_days=60, eta=0.3):
    if vix_data_series is None or vix_data_series.empty:
        st.warning("PSI: VIX data missing, using default.")
        return create_default_series(index_ref, 1.0, name="Psi_Default")
    vix_aligned = vix_data_series.reindex(index_ref).ffill().bfill()
    if vix_aligned.isnull().all():
        st.warning("PSI: VIX data all NaN after alignment, using default.")
        return create_default_series(index_ref, 1.0, name="Psi_Default_NaN_Align")
    vix_rolling_avg = vix_aligned.rolling(window=lookback_days, min_periods=max(1, lookback_days//3)).mean()
    vix_rolling_avg = vix_rolling_avg.replace(0, np.nan).ffill().bfill() 
    if vix_rolling_avg.isnull().all(): 
        st.warning("PSI: VIX rolling average all NaN, using default.")
        return create_default_series(index_ref, 1.0, name="Psi_Default_NaN_RollAvg")
    vix_normalized = (vix_aligned / vix_rolling_avg) - 1
    psi_dynamic = 1 + np.tanh(vix_normalized.fillna(0)) * eta
    return psi_dynamic.reindex(index_ref).fillna(1.0)

def calculate_adaptive_weights(price_data_df, base_weights_dict, sentiment_strength_param=0.1, momentum_lookback=20):
    index_ref = price_data_df.index if not price_data_df.empty else None
    if price_data_df.empty or not base_weights_dict:
        st.warning("Adaptive Weights: Price data empty or no base weights, using defaults.")
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
    if price_data_df.empty: 
        st.warning("Dynamic Alpha: Price data empty, using defaults.")
        # Create default for expected columns if price_data_df had columns but became empty after processing
        # This case is unlikely if the initial check passes, but good for robustness
        expected_tickers = price_data_df.columns if hasattr(price_data_df, 'columns') and price_data_df.columns.any() else []
        return {ticker: create_default_series(index_ref, 1.0, name=f"Alpha_{ticker}") for ticker in expected_tickers}

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
        st.warning("Dynamic Beta: Input data missing or empty, using default.")
        return create_default_series(index_ref, 1.0, name="Beta_Default_Initial_Empty")
    
    benchmark_aligned = benchmark_data_series.reindex(index_ref).ffill().bfill()
    component_aligned = component_data_df.reindex(index_ref).ffill().bfill() # component_data_df is price_data_df

    is_benchmark_problematic = benchmark_aligned.isnull().all()
    is_component_problematic = component_aligned.empty or (isinstance(component_aligned, pd.DataFrame) and component_aligned.isnull().all().all())
    
    if is_benchmark_problematic or is_component_problematic:
        st.warning("Dynamic Beta: Aligned benchmark or component data problematic, using default.")
        return create_default_series(index_ref, 1.0, name="Beta_Default_NaN_Align")

    benchmark_returns = benchmark_aligned.pct_change().fillna(0)
    # Ensure component_avg_returns is a Series
    if component_aligned.ndim > 1: # DataFrame
        component_avg_returns = component_aligned.pct_change().mean(axis=1).fillna(0)
    else: # Series
        component_avg_returns = component_aligned.pct_change().fillna(0)


    if (benchmark_returns == 0).all():
        st.warning("Dynamic Beta: Benchmark returns are all zero, using default beta of 1.")
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
        st.warning("Theta: Market indicators missing, using default.")
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
        st.warning("Phi: Price data empty, using default.")
        return create_default_series(index_ref, 0.0, name="Phi_Default")
    phi_total_series = create_default_series(index_ref, 0.0)
    if market_indicators_df is not None and 'VIX' in market_indicators_df.columns and not market_indicators_df['VIX'].isnull().all():
        vix_aligned = market_indicators_df['VIX'].reindex(index_ref).ffill().bfill()
        vix_momentum = vix_aligned.pct_change(3).fillna(0)
        phi_total_series = phi_total_series.add(-vix_momentum * 0.05, fill_value=0)
    
    if not price_data_df.empty and price_data_df.ndim > 1 and not price_data_df.isnull().all().all(): # Check if DataFrame and not all NaN
        volume_proxy_series = price_data_df.pct_change().abs().mean(axis=1).fillna(0)
        insider_signal_series = (volume_proxy_series - volume_proxy_series.rolling(20, min_periods=1).mean().fillna(0)) * 0.02
        phi_total_series = phi_total_series.add(insider_signal_series.reindex(index_ref).fillna(0), fill_value=0)
    
    dark_pool_signal_series = pd.Series(np.random.normal(0, 0.005, len(index_ref)), index=index_ref)
    phi_total_series = phi_total_series.add(dark_pool_signal_series, fill_value=0)
    return phi_total_series.reindex(index_ref).fillna(0.0)

def calculate_ai_forecast_omega(price_data_df, market_indicators_df, index_ref, forecast_str_param=0.05):
    if price_data_df.empty:
        st.warning("Omega: Price data empty, using default.")
        return create_default_series(index_ref, 0.0, name="Omega_Default")
    omega_components_series = create_default_series(index_ref, 0.0)
    if market_indicators_df is not None and not market_indicators_df.empty:
        mi_aligned = market_indicators_df.reindex(index_ref).ffill().bfill()
        if not mi_aligned.empty and not mi_aligned.isnull().all().all(): 
            economic_composite = mi_aligned.mean(axis=1, skipna=True).pct_change(5).fillna(0)
            omega_components_series = omega_components_series.add(economic_composite * 0.1, fill_value=0)

    satellite_signal_series = pd.Series(np.random.normal(0, 0.002, len(index_ref)), index=index_ref)
    omega_components_series = omega_components_series.add(satellite_signal_series * 0.1, fill_value=0)

    if not price_data_df.empty and price_data_df.ndim > 1 and not price_data_df.isnull().all().all():
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
        st.error("AKI: Component price data is empty. Cannot proceed.")
        return pd.Series(dtype='float64'), "AKI (No Component Data)", {}
    
    common_index = price_data_df.index
    if common_index.empty:
        st.error("Error: Price data index is empty for AKI. Cannot proceed.")
        return pd.Series(dtype='float64'), "AKI (Empty Index)", {}

    # st.info("AKI: Normalizing data...")
    normalized_components_df = robust_normalize(price_data_df)
    normalized_benchmark_series = robust_normalize(benchmark_data_series)

    # Ensure they are not empty after normalization and reindex
    if normalized_components_df.empty and not price_data_df.empty:
        st.error("Error: Normalized components are empty after robust_normalize. Aborting AKI calculation.")
        return pd.Series(dtype='float64'), "AKI (Norm Error Comp)", {}

    normalized_components_df = normalized_components_df.reindex(common_index).ffill().bfill()
    normalized_benchmark_series = normalized_benchmark_series.reindex(common_index).ffill().bfill()
    
    if normalized_benchmark_series.empty and benchmark_data_series is not None and not benchmark_data_series.empty:
        st.warning("Warning: Normalized benchmark is empty, using zeros.")
        normalized_benchmark_series = create_default_series(common_index, 0.0, name="Norm_Bench_Default")

    # st.info("AKI: Calculating dynamic components...")
    vix_series_input = market_indicators_df.get('VIX', create_default_series(common_index, 20, name="VIX_Default_Input"))
    psi_series = calculate_dynamic_psi(vix_series_input, common_index, eta=psi_eta_ui_val)
    adaptive_weights_map = calculate_adaptive_weights(price_data_df, base_weights_dict, sentiment_strength_param=momentum_strength_ui_val)
    dynamic_alphas_map = calculate_dynamic_alpha(price_data_df, options_proxy_strength_param=momentum_strength_ui_val)
    beta_series = calculate_dynamic_beta(benchmark_data_series, price_data_df, common_index)
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
            
            # Ensure norm_comp_ticker is numeric before power operation
            norm_comp_ticker = pd.to_numeric(norm_comp_ticker, errors='coerce')
            alpha_s = pd.to_numeric(alpha_s, errors='coerce')

            # Handle potential NaNs from coercion
            valid_idx = norm_comp_ticker.notna() & alpha_s.notna()
            if not valid_idx.any(): continue

            performance_ratio_s = norm_comp_ticker.loc[valid_idx] / 100.0
            
            # Element-wise power operation
            powered_performance_s = pd.Series(np.nan, index=norm_comp_ticker.index, dtype='float64')
            try:
                # Check for non-positive bases with non-integer exponents if that's a concern
                # For simplicity here, direct power
                powered_performance_s.loc[valid_idx] = performance_ratio_s.pow(alpha_s.loc[valid_idx])
            except Exception as e:
                st.warning(f"Power calculation error for {ticker}: {e}. Skipping this part for the ticker.")
                powered_performance_s.loc[valid_idx] = performance_ratio_s # fallback or handle as NaN

            weighted_performance_s = weight_s * powered_performance_s
            asset_performance_sum = asset_performance_sum.add(weighted_performance_s.fillna(0), fill_value=0)
    
    benchmark_ratio_s = (normalized_benchmark_series / 100.0) if not normalized_benchmark_series.empty else create_default_series(common_index, 0.0, name="BenchRatio_Default")
    benchmark_term = beta_s * benchmark_ratio_s
    
    # st.info("AKI: Assembling final series...")
    aki_series_final = (psi_s * asset_performance_sum) - benchmark_term + theta_s + phi_s + omega_s
    
    components_for_plot_final = {
        'psi': psi_s, 'beta': beta_s, 'theta': theta_s, 
        'phi': phi_s, 'omega': omega_s,
        'adaptive_weights': adaptive_weights_map, 
        'dynamic_alphas': dynamic_alphas_map    
    }
    return aki_series_final.dropna(), "Graduate-Level Dynamic AKI", components_for_plot_final


# --- Streamlit UI and App Logic ---
st.set_page_config(layout="wide")
st.title("🎓 AlphaKnaut Index (AKI): Graduate Edition")

TECH_TICKERS_MAP = {
    'AAPL': 'Apple Inc.', 'MSFT': 'Microsoft Corp.', 'GOOGL': 'Alphabet Inc.', 'AMZN': 'Amazon.com Inc.', 
    'NVDA': 'NVIDIA Corporation', 'META': 'Meta Platforms Inc.', 'TSLA': 'Tesla Inc.'}

# --- Sidebar for Inputs ---
st.sidebar.header("Configuration")

default_start_date = date.today() - timedelta(days=365) # Default to 1 year ago
default_end_date = date.today() - timedelta(days=1)

start_date = st.sidebar.date_input('Start Date:', value=pd.to_datetime("2023-01-01"), min_value=date(2010,1,1), max_value=default_end_date)
end_date = st.sidebar.date_input('End Date:', value=default_end_date, min_value=start_date, max_value=default_end_date)

ticker_display_options = [f"{name} ({ticker})" for ticker, name in TECH_TICKERS_MAP.items()]
selected_tickers_display = st.sidebar.multiselect('Select Stocks:', options=ticker_display_options, default=[f"{TECH_TICKERS_MAP['AAPL']} (AAPL)", f"{TECH_TICKERS_MAP['MSFT']} (MSFT)", f"{TECH_TICKERS_MAP['NVDA']} (NVDA)"])
# Extract actual tickers
selected_tickers = [opt.split('(')[-1][:-1] for opt in selected_tickers_display]


st.sidebar.subheader("Tune Dynamic Component Parameters:")
enable_advanced_tuning = st.sidebar.checkbox('Enable Advanced Tuning', value=True)

psi_eta_val = 0.3
momentum_strength_val = 0.1
forecast_strength_val = 0.05

if enable_advanced_tuning:
    psi_eta_val = st.sidebar.number_input('Ψ Sensitivity (η):', min_value=0.0, max_value=1.0, value=0.3, step=0.05, format="%.2f")
    momentum_strength_val = st.sidebar.number_input('ω/α Mom. Factor:', min_value=0.0, max_value=1.0, value=0.1, step=0.02, format="%.2f")
    forecast_strength_val = st.sidebar.number_input('Ω AI Factor:', min_value=0.0, max_value=0.5, value=0.05, step=0.01, format="%.2f")

run_button = st.sidebar.button("Run AKI Analysis", type="primary")

# --- Main Area for Output ---
if run_button:
    if not selected_tickers:
        st.error("Please select at least one stock.")
    elif start_date >= end_date:
        st.error("Start date must be before end date.")
    else:
        st.info(f"🚀 Initializing AKI Analysis for {', '.join(selected_tickers)} from {start_date} to {end_date}...")
        if enable_advanced_tuning:
            st.info(f"Parameters: η={psi_eta_val}, MomStr={momentum_strength_val}, ForecastStr={forecast_strength_val}")
        else:
            st.info("Using default dynamic parameters.")

        # Convert date objects to string for yfinance
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = (end_date + timedelta(days=1)).strftime('%Y-%m-%d') # yfinance end is exclusive

        all_fetch_tickers = list(selected_tickers) + ['^GSPC'] # Add S&P 500 as benchmark
        
        # Progress bar for data fetching (visual only as yf doesn't expose progress well)
        progress_bar = st.progress(0, text="Fetching data...")

        price_data_df, market_indicators_df = get_enhanced_market_data(
            all_fetch_tickers, start_date_str, end_date_str
        )
        progress_bar.progress(30, text="Data fetched. Processing...")

        if price_data_df.empty:
            st.error("❌ No price data downloaded. Analysis aborted.")
            st.stop() # Stop execution of the script here for this run
        
        valid_selected_tickers = [t for t in selected_tickers if t in price_data_df.columns]
        if not valid_selected_tickers:
            st.error(f"❌ None of the selected stocks ({', '.join(selected_tickers)}) found in downloaded price data. Available: {', '.join(price_data_df.columns)}. Analysis aborted.")
            st.stop()
        
        component_data_df = price_data_df[valid_selected_tickers]
        benchmark_data_series = price_data_df.get('^GSPC', pd.Series(dtype='float64'))
        
        if benchmark_data_series.empty and '^GSPC' in price_data_df.columns : # Check if it was downloaded but became empty
             benchmark_data_series = price_data_df['^GSPC'] # Try to re-assign if it was present
        elif benchmark_data_series.empty:
             st.warning("S&P 500 (^GSPC) benchmark data not found or empty. Beta calculations might be affected or use defaults.")
             # Create a dummy series if it's completely missing, to avoid errors downstream, though beta will be default
             if not component_data_df.empty:
                 benchmark_data_series = pd.Series(100, index=component_data_df.index, name='^GSPC_dummy')


        if component_data_df.empty:
            st.error("❌ No component data after filtering. Analysis aborted.")
            st.stop()

        base_weights_dict = {ticker: 1.0/len(valid_selected_tickers) for ticker in valid_selected_tickers}
        
        progress_bar.progress(50, text="Calculating AKI components...")
        aki_series, title, components_dict = create_graduate_level_aki(
            component_data_df, benchmark_data_series, market_indicators_df, base_weights_dict,
            psi_eta_ui_val=psi_eta_val, 
            momentum_strength_ui_val=momentum_strength_val, 
            forecast_strength_ui_val=forecast_strength_val
        )
        progress_bar.progress(80, text="Components calculated. Plotting...")
            
        if aki_series.empty:
            st.error("❌ AKI calculation resulted in empty series. Cannot plot.")
            st.stop()

        st.subheader(title)
        
        fig = make_subplots(
            rows=3, cols=2, column_widths=[0.5, 0.5], row_heights=[0.4, 0.3, 0.3],
            subplot_titles=['AKI vs S&P 500 (Normalized)', 'Ψ Market Regime', 'β Dynamic Beta', 
                            'Θ Cross-Asset', 'Φ Behavioral', 'Ω AI Forecast'], # Adjusted titles
            specs=[[{"colspan": 2, "secondary_y":True}, None], 
                   [{}, {}], 
                   [{}, {}]]
        )
        fig.add_trace(go.Scatter(x=aki_series.index, y=aki_series, name='AKI', line=dict(width=2, color='blue')), 
                      secondary_y=False, row=1, col=1)
        
        if benchmark_data_series is not None and not benchmark_data_series.empty:
            norm_bench_series = robust_normalize(benchmark_data_series) 
            if not norm_bench_series.empty:
                fig.add_trace(go.Scatter(x=norm_bench_series.index, y=norm_bench_series, name='S&P 500 (Norm)', line=dict(dash='dash', color='rgba(128,128,128,0.7)')),
                              secondary_y=True, row=1, col=1)
        
        # Plot Psi, Beta, Theta, Phi, Omega
        plot_details = [
            ('psi', 'Ψ Market Regime', 2, 1), 
            ('beta', 'β Dynamic Beta', 2, 2),
            ('theta', 'Θ Cross-Asset', 3, 1),
            ('phi', 'Φ Behavioral', 3, 2), # Moved Phi here, original code had it commented or implicitly on omega's plot
            ('omega', 'Ω AI Forecast', 3, 2) # Omega will overlay Phi now as per this sequence. Or adjust subplot_titles
        ]
        # Let's ensure Phi and Omega have distinct places or adjust subplot titles
        # For distinct plots:
        # Option 1: New row (4 rows total)
        # Option 2: 3 columns in one row (if space allows)
        # For now, will plot them in the designated spots, Phi in (3,2), Omega in (3,2) - will result in overlay
        # Corrected subplot titles and mapping:
        # Subplot titles are: 'AKI vs S&P 500', 'Ψ Market Regime', 'β Dynamic Beta', 'Θ Cross-Asset', 'Φ Behavioral', 'Ω AI Forecast'
        # This implies (2,1) for Psi, (2,2) for Beta, (3,1) for Theta, (3,2) for Phi, (NEW PLOT) for Omega
        # Let's adjust layout for all 5 components below main AKI plot
        
        # NEW LAYOUT FOR COMPONENTS:
        fig = make_subplots(
            rows=4, cols=2, # Increased rows to accommodate all components separately if needed
            column_widths=[0.5, 0.5], row_heights=[0.4, 0.2, 0.2, 0.2], # Adjusted row heights
            subplot_titles=['AKI vs S&P 500 (Normalized)', 
                            'Ψ Market Regime (Psi)', 'β Dynamic Beta', 
                            'Θ Cross-Asset (Theta)', 'Φ Behavioral (Phi)',
                            'Ω AI Forecast (Omega)', None], # Added one more title spot, though Omega can go to (4,1) or (4,2)
            specs=[[{"colspan": 2, "secondary_y":True}, None], 
                   [{}, {}], 
                   [{}, {}],
                   [{}, {}]] # Added spec for 4th row
        )
        fig.add_trace(go.Scatter(x=aki_series.index, y=aki_series, name='AKI', line=dict(width=2, color='blue')), 
                      secondary_y=False, row=1, col=1)
        
        if benchmark_data_series is not None and not benchmark_data_series.empty:
            norm_bench_series = robust_normalize(benchmark_data_series.dropna()) # Dropna before normalize
            if not norm_bench_series.empty:
                fig.add_trace(go.Scatter(x=norm_bench_series.index, y=norm_bench_series, name='S&P 500 (Norm)', line=dict(dash='dash', color='rgba(128,128,128,0.7)')),
                              secondary_y=True, row=1, col=1)

        plot_map = {
            'psi': (2,1), 'beta': (2,2),
            'theta': (3,1), 'phi': (3,2),
            'omega': (4,1) # Omega gets its own spot
        }
        
        for comp_key, (r, c) in plot_map.items():
            comp_data_series = components_dict.get(comp_key)
            if comp_data_series is not None and not comp_data_series.empty:
                 fig.add_trace(go.Scatter(x=comp_data_series.index, y=comp_data_series, name=comp_key.upper()), row=r, col=c)
        
        fig.update_layout(height=1200, title_text=None, showlegend=True, # Main title already set by st.title
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        fig.update_yaxes(title_text='AKI Value', secondary_y=False, row=1, col=1)
        fig.update_yaxes(title_text='S&P 500 Norm.', secondary_y=True, showgrid=False, row=1, col=1)
        
        for r_idx in [2,3,4]: 
            for c_idx in [1,2]:
                 try: # subplot might not exist if using colspan
                     fig.update_yaxes(title_text="Factor Value", row=r_idx, col=c_idx, title_font=dict(size=10), tickfont=dict(size=9))
                 except ValueError:
                     pass # Subplot does not exist

        st.plotly_chart(fig, use_container_width=True)
        progress_bar.progress(100, text="Analysis Complete!")
        progress_bar.empty() # Remove progress bar
            
        if not aki_series.empty:
            st.subheader(f"AKI Summary ({start_date} to {end_date})")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Start AKI", f"{aki_series.iloc[0]:.3f}")
            col2.metric("End AKI", f"{aki_series.iloc[-1]:.3f}")
            col3.metric("AKI Min", f"{aki_series.min():.3f}")
            col4.metric("AKI Max", f"{aki_series.max():.3f}")
            # st.write(f"   AKI Std Dev: {aki_series.std():.3f}")
            st.dataframe(aki_series.to_frame(name="AKI").tail())

else:
    st.info("Adjust parameters in the sidebar and click 'Run AKI Analysis' to start.")