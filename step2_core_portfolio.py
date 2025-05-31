import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings('ignore')

def create_default_series(index_ref, default_val=0.0, name=None):
    if index_ref is None or len(index_ref) == 0:
        index_ref = pd.date_range(start='2020-01-01', periods=10, freq='B')
    return pd.Series(default_val, index=index_ref, name=name)

def robust_normalize(data_series_or_df):
    if data_series_or_df is None or data_series_or_df.empty:
        return pd.Series(dtype='float64') if isinstance(data_series_or_df, pd.Series) else pd.DataFrame()
    data_copy = data_series_or_df.copy()
    first_valid_row_values = data_copy.ffill().iloc[0]
    if isinstance(data_copy, pd.DataFrame):
        normalized_cols = {}
        for col in data_copy.columns:
            col_series = data_copy[col]
            first_val_col = col_series.ffill().iloc[0]
            if pd.isna(first_val_col) or first_val_col == 0:
                first_valid_nonzero_col = None
                for val in col_series.ffill().bfill():
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
        first_val_series = first_valid_row_values
        if pd.isna(first_val_series) or first_val_series == 0:
            first_valid_nonzero_series = None
            for val in data_copy.ffill().bfill():
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
        for indicator_yf, indicator_name in market_indicators_map.items():
            if indicator_name not in market_df.columns or market_df[indicator_name].isnull().all():
                print(f"Warning: {indicator_name} data missing or empty, using synthetic {indicator_name}.")
                synthetic_series = None
                if indicator_name == 'VIX':
                    synthetic_series = pd.Series(np.random.normal(20, 5, len(base_index_for_synthetic)), index=base_index_for_synthetic).clip(10, 50)
                elif indicator_name == 'TNX':
                    synthetic_series = pd.Series(np.random.normal(2.5, 0.5, len(base_index_for_synthetic)), index=base_index_for_synthetic).clip(0.5, 5)
                elif indicator_name == 'DXY':
                    synthetic_series = pd.Series(np.random.normal(100, 3, len(base_index_for_synthetic)), index=base_index_for_synthetic).clip(90, 110)
                if synthetic_series is not None:
                    market_df[indicator_name] = synthetic_series
        if not price_df.empty:
            price_df = price_df.ffill().bfill().dropna(how='all')
        if not market_df.empty:
            market_df = market_df.ffill().bfill().dropna(how='all')
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
            market_df = pd.DataFrame(index=price_df.index, columns=list(market_indicators_map.values()))
        return price_df, market_df
    except Exception as e:
        print(f"CRITICAL Error during enhanced download: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame(), pd.DataFrame()

def main():
    # Example: Magnificent 7
    tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META', 'NVDA', 'TSLA']
    start_date = '2020-01-01'
    end_date = '2025-01-01'
    price_df, market_df = get_enhanced_market_data(tickers, start_date, end_date)
    if price_df.empty:
        print("No price data available.")
        return
    norm_prices = robust_normalize(price_df)
    # Example index: equally weighted mean of normalized prices
    aki_index = norm_prices.mean(axis=1)
    # Plot
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Adaptive Index", "Normalized Prices"))
    fig.add_trace(go.Scatter(x=aki_index.index, y=aki_index, name="AKI Index"), row=1, col=1)
    for col in norm_prices.columns:
        fig.add_trace(go.Scatter(x=norm_prices.index, y=norm_prices[col], name=col, showlegend=False), row=2, col=1)
    fig.update_layout(title="Adaptive Index & Magnificent 7 Normalized Prices", height=800)
    fig.show()

if __name__ == "__main__":
    main()
