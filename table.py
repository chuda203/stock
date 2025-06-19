import streamlit as st
import pandas as pd

def display_fitting_table(stock_symbol, fitting_dates, closing_prices, Fitting_S_n_list):
    """
    Display a table for the fitting plot showing dates, actual prices, and fitted prices.
    """
    st.subheader(f"📋 Data Table for Fitting Plot ({stock_symbol})")
    
    # Create DataFrame for the fitting data
    fitting_df = pd.DataFrame({
        'Date': [d.strftime('%Y-%m-%d') for d in fitting_dates],  # Format dates as strings
        'Actual Price': closing_prices,
        'Fitted Price': Fitting_S_n_list[:len(fitting_dates)]
    })
    
    # Display the table in Streamlit
    st.dataframe(
        fitting_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            'Date': st.column_config.TextColumn('Date'),
            'Actual Price': st.column_config.NumberColumn('Actual Price', format="%.2f"),
            'Fitted Price': st.column_config.NumberColumn('Fitted Price', format="%.2f")
        }
    )

def display_fitting_forecast_table(stock_symbol, fitting_dates, closing_prices, Fitting_S_n_list,
                                  forecast_dates, S_forecast, actual_forecast_prices):
    """
    Display a table for the fitting + forecast plot showing dates, actual prices, fitted prices,
    and forecast prices.
    """
    st.subheader(f"📋 Data Table for Fitting + Forecast Plot ({stock_symbol})")
    
    # Create fitting DataFrame
    fitting_df = pd.DataFrame({
        'Date': [d.strftime('%Y-%m-%d') for d in fitting_dates],
        'Actual Price': closing_prices,
        'Fitted Price': Fitting_S_n_list[:len(fitting_dates)],
        'Forecast Price': [None] * len(fitting_dates),
        'Type': ['Fitting'] * len(fitting_dates)
    })
    
    # Create forecast DataFrame
    min_len = min(len(forecast_dates), len(S_forecast), len(actual_forecast_prices))
    forecast_df = pd.DataFrame({
        'Date': [d.strftime('%Y-%m-%d') for d in forecast_dates[:min_len]],
        'Actual Price': actual_forecast_prices[:min_len],
        'Fitted Price': [None] * min_len,
        'Forecast Price': S_forecast[:min_len],
        'Type': ['Forecast'] * min_len
    })
    
    # Combine DataFrames
    combined_df = pd.concat([fitting_df, forecast_df], ignore_index=True)
    
    # Display the table in Streamlit
    st.dataframe(
        combined_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            'Date': st.column_config.TextColumn('Date'),
            'Actual Price': st.column_config.NumberColumn('Actual Price', format="%.2f"),
            'Fitted Price': st.column_config.NumberColumn('Fitted Price', format="%.2f"),
            'Forecast Price': st.column_config.NumberColumn('Forecast Price', format="%.2f"),
            'Type': st.column_config.TextColumn('Type')
        }
    )

def display_mape_table(stock_symbol, mape_data, period_type):
    """
    Display a table for the MAPE plot showing day index and MAPE values.
    """
    st.subheader(f"📋 Data Table for MAPE {period_type} Plot ({stock_symbol})")
    
    # Create DataFrame for MAPE data
    mape_df = pd.DataFrame({
        'Day': range(1, len(mape_data) + 1),
        f'MAPE {period_type} (%)': mape_data
    })
    
    # Display the table in Streamlit
    st.dataframe(
        mape_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            'Day': st.column_config.NumberColumn('Day'),
            f'MAPE {period_type} (%)': st.column_config.NumberColumn(f'MAPE {period_type} (%)', format="%.2f")
        }
    )

def display_raw_data_table(stock_symbol, fitting_data, forecast_data, start_date, end_date, forecast_end_date):
    """
    Display a table for the raw Yahoo Finance data for the specified stock symbol and date range.
    """
    st.subheader(f"📋 Raw Data from Yahoo Finance ({stock_symbol})")
    st.markdown(f"Data retrieved for the period: {start_date.strftime('%d/%m/%Y')} to {forecast_end_date.strftime('%d/%m/%Y')}")
    
    # Combine fitting and forecast data
    if forecast_data is not None and not forecast_data.empty:
        combined_data = pd.concat([fitting_data, forecast_data])
    else:
        combined_data = fitting_data
    
    # Ensure the data is sorted by date
    combined_data = combined_data.sort_index()
    
    # Reset index to make 'Date' a column and format it as string
    raw_data_display = combined_data.reset_index()
    raw_data_display['Date'] = raw_data_display['Date'].dt.strftime('%Y-%m-%d')
    
    # Rename columns to include stock symbol
    raw_data_display = raw_data_display.rename(columns={
        'Open': f'Open ({stock_symbol})',
        'High': f'High ({stock_symbol})',
        'Low': f'Low ({stock_symbol})',
        'Close': f'Close ({stock_symbol})',
        'Adj Close': f'Adj Close ({stock_symbol})',
        'Volume': f'Volume ({stock_symbol})'
    })
    
    # Display the table in Streamlit
    st.dataframe(
        raw_data_display,
        use_container_width=True,
        # Remove hide_index=True to show row numbers
        column_config={
            "Date": st.column_config.DateColumn("Date"),
            f"Open ({stock_symbol})": st.column_config.NumberColumn(f"Open ({stock_symbol})", format="%.2f"),
            f"High ({stock_symbol})": st.column_config.NumberColumn(f"High ({stock_symbol})", format="%.2f"),
            f"Low ({stock_symbol})": st.column_config.NumberColumn(f"Low ({stock_symbol})", format="%.2f"),
            f"Close ({stock_symbol})": st.column_config.NumberColumn(f"Close ({stock_symbol})", format="%.2f"),
            f"Adj Close ({stock_symbol})": st.column_config.NumberColumn(f"Adj Close ({stock_symbol})", format="%.2f"),
            f"Volume ({stock_symbol})": st.column_config.NumberColumn(f"Volume ({stock_symbol})", format="%d")
        }
    )