import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from table import display_fitting_table, display_fitting_forecast_table, display_mape_table

def plot_fitting(stock_symbol, fitting_dates, closing_prices, Fitting_S_n_list):
    st.subheader(f"📊 Grafik Fitting vs Actual ({stock_symbol})")
    fig_fit, ax_fit = plt.subplots(figsize=(12, 6))
    
    # Plot with dates on x-axis
    ax_fit.plot(fitting_dates, closing_prices, label="Actual", color='black', linewidth=2)
    ax_fit.plot(fitting_dates, Fitting_S_n_list[:len(fitting_dates)], label="Fitted", color='blue', linewidth=2)
    
    ax_fit.set_title(f"Fitting Data Harga Saham ({stock_symbol})")
    ax_fit.set_xlabel("Tanggal")
    ax_fit.set_ylabel("Harga")
    ax_fit.legend()
    ax_fit.grid(True, alpha=0.3)
    
    # Format x-axis dates
    ax_fit.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    
    st.pyplot(fig_fit)
    
    # Display table for fitting data
    display_fitting_table(stock_symbol, fitting_dates, closing_prices, Fitting_S_n_list)

def plot_fitting_forecast(stock_symbol, fitting_dates, closing_prices, Fitting_S_n_list, 
                         forecast_dates, S_forecast, actual_forecast_prices):
    st.subheader(f"📈 Grafik Fitting + Forecast vs Actual ({stock_symbol})")
    fig_forecast, ax_forecast = plt.subplots(figsize=(14, 7))
    
    # Plot fitting period
    ax_forecast.plot(fitting_dates, closing_prices, label="Actual (Fitting)", color='black', linewidth=2)
    ax_forecast.plot(fitting_dates, Fitting_S_n_list[:len(fitting_dates)], label="Fitted", color='blue', linewidth=2)
    
    # Plot forecast period
    ax_forecast.plot(forecast_dates, actual_forecast_prices, label="Actual (Forecast)", color='darkgreen', linewidth=2)
    ax_forecast.plot(forecast_dates, S_forecast[:len(forecast_dates)], label="Forecast", color='orange', linewidth=2)
    
    if fitting_dates and forecast_dates and len(Fitting_S_n_list) > 0 and len(S_forecast) > 0:
        last_fitting_date = fitting_dates[-1]
        last_fitting_price = Fitting_S_n_list[len(fitting_dates)-1]
        
        first_forecast_date = forecast_dates[0]
        first_forecast_price = S_forecast[0]
        
        ax_forecast.plot([last_fitting_date, first_forecast_date], 
                        [last_fitting_price, first_forecast_price], 
                        color='orange', linewidth=2, linestyle='-')
    
    if fitting_dates and forecast_dates and len(closing_prices) > 0 and len(actual_forecast_prices) > 0:
        last_actual_fitting_date = fitting_dates[-1]
        last_actual_fitting_price = closing_prices[-1]
        
        first_actual_forecast_date = forecast_dates[0]
        first_actual_forecast_price = actual_forecast_prices[0]
        
        ax_forecast.plot([last_actual_fitting_date, first_actual_forecast_date], 
                        [last_actual_fitting_price, first_actual_forecast_price], 
                        color='darkgreen', linewidth=2, linestyle='-')
    
    # Add vertical line to separate fitting and forecast
    if fitting_dates and forecast_dates:
        ax_forecast.axvline(x=fitting_dates[-1], color='red', linestyle='--', 
                           label='Forecast Start', alpha=0.7)
    
    ax_forecast.set_title(f"Fitting dan Forecast Harga Saham ({stock_symbol})")
    ax_forecast.set_xlabel("Tanggal")
    ax_forecast.set_ylabel("Harga")
    ax_forecast.legend()
    ax_forecast.grid(True, alpha=0.3)
    
    # Format x-axis dates
    ax_forecast.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    
    st.pyplot(fig_forecast)
    
    # Display table for fitting + forecast data
    display_fitting_forecast_table(stock_symbol, fitting_dates, closing_prices, Fitting_S_n_list,
                                  forecast_dates, S_forecast, actual_forecast_prices)
        
def plot_mape(stock_symbol, mape_data, period_type, mean_mape):
    st.subheader(f"📉 Hasil MAPE {period_type} - Rata-rata: {mean_mape:.2f}%")
    fig_mape, ax_mape = plt.subplots(figsize=(10, 6))
    ax_mape.plot(mape_data, color='purple' if period_type == "Fitting" else 'orange',
                label=f'MAPE {period_type} (%)', linewidth=2)
    ax_mape.set_title(f"Grafik MAPE Selama {period_type} ({stock_symbol})")
    ax_mape.set_xlabel("Hari")
    ax_mape.set_ylabel("MAPE (%)")
    ax_mape.legend()
    ax_mape.grid(True, alpha=0.3)
    st.pyplot(fig_mape)
    
    # Display table for MAPE data
    display_mape_table(stock_symbol, mape_data, period_type)