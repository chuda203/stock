import streamlit as st
from datetime import datetime, timedelta

def create_ui():
    st.title("📈 Stock Price Fitting and Forecasting Web")
    st.markdown("---")
    
    # Input Parameter
    st.subheader("📋 Input Parameter")
    
    # Create 4 columns for the input fields
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("**Stock Symbol**")
        stock_symbol = st.text_input("", value="BBCA.JK", key="stock_input", label_visibility="collapsed")
    
    with col2:
        st.markdown("**Fitting Start Date**")
        start_date = st.date_input("", value=datetime(2024, 1, 1), key="start_date_input", label_visibility="collapsed")
    
    with col3:
        st.markdown("**Fitting Period (Day)**")
        training_days = st.number_input("", min_value=1, max_value=365, value=120, step=1, key="training_days", label_visibility="collapsed")
    
    with col4:
        st.markdown("**Forecast Period (Day)**")
        forecast_days = st.number_input("", min_value=1, max_value=365, value=60, step=1, key="forecast_days", label_visibility="collapsed")
    
    # Calculate dates based on training and forecast days
    end_date = start_date + timedelta(days=training_days)
    forecast_end_date = end_date + timedelta(days=forecast_days)
    
    # Advanced Options (Collapsible)
    with st.expander("⚙️ Advanced Options"):
        col_adv1, col_adv2 = st.columns(2)
        
        with col_adv1:
            use_custom_end = st.checkbox("Custom Fitting End Date", value=False)
            if use_custom_end:
                st.markdown("**Custom Fitting End Date**")
                custom_end_date = st.date_input("", value=datetime(2024, 4, 30), key="custom_end", label_visibility="collapsed")
                end_date = custom_end_date
        
        with col_adv2:
            use_custom_forecast_end = st.checkbox("Custom Forecast End Date", value=False)
            if use_custom_forecast_end:
                st.markdown("**Custom Forecast End Date**")
                custom_forecast_end = st.date_input("", value=datetime(2024, 6, 29), key="custom_forecast_end", label_visibility="collapsed")
                forecast_end_date = custom_forecast_end
    
    # Ringkasan Input
    col_summary1, col_summary2 = st.columns(2)
    
    with col_summary1:
        st.markdown(f"""
        **📊 Fitting Period Details:**
        - **Periode Fitting:** {start_date.strftime('%d/%m/%Y')} - {end_date.strftime('%d/%m/%Y')}
        - **Durasi Fitting:** {training_days} hari
        """)
    
    with col_summary2:
        st.markdown(f"""
        **🔮 Data Forecast Period Details:**
        - **Periode Forecast:** {end_date.strftime('%d/%m/%Y')} - {forecast_end_date.strftime('%d/%m/%Y')}
        - **Durasi Forecast:** {forecast_days} hari
        """)
    
    st.markdown("---")
    
    return stock_symbol, start_date, training_days, forecast_days, end_date, forecast_end_date