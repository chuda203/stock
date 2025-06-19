import pandas as pd
import yfinance as yf
import logging

def get_data_with_dates(stock_name, start_date, end_date, forecast_end_date):
    """
    Get stock data with proper date alignment for both fitting and forecasting
    """
    try:
        # Download all data from start_date to forecast_end_date
        all_data = yf.download(stock_name, start=start_date, end=forecast_end_date, auto_adjust=False)
        
        if all_data.empty:
            logging.error(f"No data available for {stock_name}")
            return None, None
        
        # Handle multi-ticker data
        if isinstance(all_data.columns, pd.MultiIndex):
            logging.warning(f"Multi-ticker data detected for {stock_name}. Selecting first ticker.")
            all_data = all_data.xs(stock_name, axis=1, level=1, drop_level=True)
        
        # Split into fitting and forecast data
        fitting_data = all_data[all_data.index < pd.Timestamp(end_date)]
        forecast_data = all_data[all_data.index >= pd.Timestamp(end_date)]
        
        logging.info(f"Fitting data: {len(fitting_data)} points from {fitting_data.index[0]} to {fitting_data.index[-1]}")
        logging.info(f"Forecast data: {len(forecast_data)} points from {forecast_data.index[0]} to {forecast_data.index[-1]}")
        
        return fitting_data, forecast_data
        
    except Exception as e:
        logging.error(f"Error getting data for {stock_name}: {e}")
        return None, None

def get_data(stock_name, start_date, end_date):
    """Legacy function for backward compatibility"""
    data = yf.download(stock_name, start=start_date, end=end_date, auto_adjust=False)
    if data.empty:
        return []
    closing_prices = data['Close']
    s_obs = closing_prices.to_numpy()
    return s_obs

def filter_prices_duplicates(data_df):
    """
    Filter duplicate prices while maintaining date alignment
    """
    if data_df is None or data_df.empty:
        return pd.DataFrame()
    
    # Remove consecutive duplicate closing prices
    filtered_data = data_df.copy()
    if isinstance(filtered_data['Close'], pd.DataFrame):
        logging.error(f"Unexpected: 'Close' column is a DataFrame: {filtered_data['Close'].head()}")
        return pd.DataFrame()
    mask = filtered_data['Close'].diff() != 0
    mask.iloc[0] = True  # Always keep first value
    
    filtered_data = filtered_data[mask]
    
    logging.info(f"Filtered data: {len(filtered_data)} points after removing {len(data_df) - len(filtered_data)} duplicates")
    
    return filtered_data
