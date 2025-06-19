import streamlit as st
from datetime import datetime, timedelta
import logging
import numpy as np
import pandas as pd
from ui import create_ui
from store import get_data_with_dates, filter_prices_duplicates
from formula import fitting, forecasting, determine_MAPE_list
from chart import plot_fitting, plot_fitting_forecast, plot_mape
from export import create_excel_download
from table import display_raw_data_table  

logging.basicConfig(
    level=logging.DEBUG, 
    format="%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s",
    handlers=[logging.StreamHandler()]
)

def main():
    # Get UI inputs
    stock_symbol, start_date, training_days, forecast_days, end_date, forecast_end_date = create_ui()

    # Validation
    if start_date >= end_date:
        st.error("Start date harus lebih kecil dari end date!")
        return
    elif end_date >= forecast_end_date:
        st.error("End date harus lebih kecil dari forecast end date!")
        return

    # Run Analysis Button
    run_forecast = st.button("🔗 Submit Data", use_container_width=True, type="primary")

    if run_forecast:
        try:
            with st.spinner("Mengambil dan memproses data..."):
                # Get data, function from store.py
                fitting_data, forecast_data = get_data_with_dates(
                    stock_symbol, start_date, end_date, forecast_end_date
                )
                
                if fitting_data is None or len(fitting_data) < 4:
                    st.error("Data tidak cukup untuk melakukan forecasting. Minimal 4 data point diperlukan.")
                    return
                if isinstance(fitting_data['Close'], pd.DataFrame):
                    st.error("Unexpected data structure: 'Close' column is a DataFrame. Please check the stock symbol or data source.")
                    logging.error(f"fitting_data['Close'] is a DataFrame: {fitting_data['Close'].head()}")
                    return
                
                logging.debug(f"fitting_data type: {type(fitting_data)}")
                logging.debug(f"fitting_data['Close'] type: {type(fitting_data['Close'])}")
                logging.debug(f"fitting_data.index type: {type(fitting_data.index)}")

                # Extract prices and dates for fitting
                fitting_prices = fitting_data['Close'].tolist()
                fitting_dates = fitting_data.index.tolist()
                
                # Filter duplicates while preserving date alignment
                filtered_data = filter_prices_duplicates(fitting_data)
                if filtered_data.empty:
                    st.error("No data remains after filtering duplicates.")
                    return
                if isinstance(filtered_data['Close'], pd.DataFrame):
                    st.error("Unexpected data structure in filtered data: 'Close' column is a DataFrame.")
                    logging.error(f"filtered_data['Close'] is a DataFrame: {filtered_data['Close'].head()}")
                    return
                fitting_prices = filtered_data['Close'].tolist()
                fitting_dates = filtered_data.index.tolist()

                # FITTING
                Fitting_S_n_list, v_list = fitting(fitting_prices, stock_symbol)
                
                if not Fitting_S_n_list:
                    st.error("Gagal melakukan fitting data.")
                    return
                    
                mape_fit = determine_MAPE_list(fitting_prices, Fitting_S_n_list)

                # FORECASTING
                S_forecast, forecast_dates, actual_forecast_prices = forecasting(
                    Fitting_S_n_list,
                    forecast_data,
                    stock_symbol
                )
                
                if S_forecast and actual_forecast_prices:
                    mape_forecast = determine_MAPE_list(actual_forecast_prices, S_forecast)
                else:
                    mape_forecast = []

            st.success("Selesai!")

            # Display raw data table
            display_raw_data_table(stock_symbol, fitting_data, forecast_data, start_date, end_date, forecast_end_date)

            # Display results
            if Fitting_S_n_list:
                # Display summary statistics
                st.subheader("📊 Statistic Details")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Jumlah Data Fitting", len(fitting_prices))
                with col2:
                    if mape_fit:
                        st.metric("MAPE Fitting", f"{np.mean(mape_fit):.2f}%")
                with col3:
                    if mape_forecast:
                        st.metric("MAPE Forecast", f"{np.mean(mape_forecast):.2f}%")
                with col4:
                    forecast_days_actual = len(S_forecast) if S_forecast else 0
                    st.metric("Periode Forecast", f"{forecast_days_actual} hari")

                # Plot charts with proper date alignment
                plot_fitting(stock_symbol, fitting_dates, fitting_prices, Fitting_S_n_list)
                
                if S_forecast and actual_forecast_prices:
                    plot_fitting_forecast(
                        stock_symbol, 
                        fitting_dates, fitting_prices, Fitting_S_n_list,
                        forecast_dates, S_forecast, actual_forecast_prices
                    )
                
                if mape_fit:
                    plot_mape(stock_symbol, mape_fit, "Fitting", np.mean(mape_fit))
                
                if mape_forecast:
                    plot_mape(stock_symbol, mape_forecast, "Forecast", np.mean(mape_forecast))

                # Create download button for Excel
                st.subheader("💾 Download Data")
                try:
                    excel_data = create_excel_download(
                        stock_symbol=stock_symbol,
                        fitting_dates=fitting_dates,
                        fitting_prices=fitting_prices,
                        Fitting_S_n_list=Fitting_S_n_list,
                        forecast_dates=forecast_dates if forecast_dates else [],
                        S_forecast=S_forecast if S_forecast else [],
                        actual_forecast_prices=actual_forecast_prices if actual_forecast_prices else []
                    )
                            
                    filename = f"{stock_symbol}_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                            
                    st.download_button(
                        label="📥 Download Excel Report",
                        data=excel_data,
                        file_name=filename,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        help="Download complete analysis data in Excel format"
                    )
                            
                    st.info(f"📊 File akan berisi data dari {start_date} hingga {forecast_end_date}")
                            
                except Exception as e:
                    st.error(f"Error creating Excel file: {str(e)}")
                    logging.error(f"Excel creation error: {e}")
                    
        except Exception as e:
            st.error(f"Terjadi kesalahan: {str(e)}")
            logging.error(f"Main execution error: {e}")
            st.info("Silakan coba dengan parameter yang berbeda atau periksa koneksi data.")

if __name__ == "__main__":
    main()