import streamlit as st
import numpy as np
import mpmath as mp
import logging
from store import get_data, filter_prices_duplicates

mp.dps = 100

def determine_v_n(Sn, Sn_1):
    v_n = (Sn - Sn_1) / 1 #delta_t = 1
    if abs(v_n) < 1e-12:
        return 1e-12 
    return v_n

def determine_alpha_n(Sn_minus_4, Sn_minus_3, Sn_minus_2, Sn_minus_1):
    AA = (Sn_minus_2 - 2 * Sn_minus_3 + Sn_minus_4)
    BB = (Sn_minus_1 - Sn_minus_2)
    CC = (Sn_minus_1 - 2 * Sn_minus_2 + Sn_minus_3)
    DD = (Sn_minus_2 - Sn_minus_3)
    alpha_pembilang = (AA * BB) - (CC * DD)
    alpha_penyebut = DD * BB * (DD - BB) 
    if abs(alpha_penyebut) < 1e-12:
        return 1e-12  
    return (alpha_pembilang/alpha_penyebut)

def determine_beta_n(Sn_minus_3, Sn_minus_2, Sn_minus_1, alpha_n):
    CC = (Sn_minus_1 - 2 * Sn_minus_2 + Sn_minus_3)
    BB = (Sn_minus_1 - Sn_minus_2)
    if abs(BB) < 1e-12:
        return 1e-12 
    return (CC-(alpha_n * (BB**2)))/(BB * 1) #delta_t = 1

def determine_h_n(v_1, alpha_n, beta_n):
    if abs(alpha_n) < 1e-12:
        alpha_n = 1e-12
    if abs(v_1) < 1e-12:
        v_1 = 1e-12
    try:
        h_n = abs((v_1 + (beta_n / alpha_n) / v_1))
        return h_n
    except (ZeroDivisionError) as e:
        logging.warning(f"Error in determine_h_n: {e}. Using fallback value.")
        return 1.0

def determine_s_n(s1, alpha, beta, h, condition_1, s_n, v_n, v_1):
    logging.debug(f"determine_s_n called with: s1={s1}, alpha={alpha}, beta={beta}, h={h}, condition_1={condition_1}, s_n={s_n}, v_n={v_n}, v_1={v_1}")
    if abs(alpha) < 1e-12:
        alpha = 1e-12
    if abs(beta) < 1e-12:
        beta = 1e-12
    condition_2 = v_n > v_1
    condition_3 = s_n > s1
    try:
        if condition_1 > 0 and condition_2 and condition_3:
            s_n = s1 - (1/alpha) * mp.log(mp.fabs((mp.exp(beta) - h) / (1 - h)))
        if condition_1 > 0 and condition_2 and not condition_3:
            s_n = s1 + mp.fabs(1/alpha) * (mp.fabs(beta)/beta) * mp.log(mp.fabs((mp.exp(beta) - h) / (1 - h)))
        if condition_1 < 0 and condition_2 and condition_3:
            s_n = s1 - (1/alpha) * mp.log(mp.fabs((mp.exp(beta) + h) / (1 + h)))
        if condition_1 < 0 and condition_2 and not condition_3:
            s_n = s1 - mp.fabs(1/alpha) * (mp.fabs(beta)/beta) * mp.log(mp.fabs((mp.exp(beta) + h) / (1 + h)))
        if condition_1 > 0 and not condition_2 and condition_3:
            s_n = s1 - (1/alpha) * (beta/mp.fabs(beta)) * mp.log(mp.fabs((mp.exp(beta) -h) / (1 - h)))
        if condition_1 > 0 and not condition_2 and not condition_3:
            s_n = s1 - mp.fabs(1/alpha) * mp.log(mp.fabs((mp.exp(-mp.fabs(beta)) - h) / (1 - h)))
        if condition_1 < 0 and not condition_2 and condition_3:
            s_n = s1 + (1/alpha) * (beta/mp.fabs(beta)) * mp.log(mp.fabs(mp.exp(-mp.fabs(beta)) + h) / (1 + h))
        if condition_1 < 0 and not condition_2 and not condition_3:
            s_n = s1 + mp.fabs(1/alpha) * mp.log(mp.fabs(mp.exp(-mp.fabs(beta)) + h) / (1 + h))
    except (ZeroDivisionError) as e:
        logging.error(f'Error in determine_s_n: {e}. Using fallback value.')
        s_n = s1
    logging.debug(f'determine_s_n result: s_n={s_n}')
    return s_n

def determine_MAPE_list(actual: list, predicted: list) -> list:
    logging.debug(f'actual: {actual}, len {len(actual)}')
    logging.debug(f"predicted: {predicted}, len {len(predicted)}")
    min_len = min(len(actual), len(predicted))
    actual = actual[:min_len]
    predicted = predicted[:min_len]
    num_of_cases = len(actual)
    sum_of_percentage_error = 0
    mape_list = []
    for i in range(num_of_cases):
        if actual[i] == 0:
            continue
        abs_error = mp.fabs(actual[i] - predicted[i])
        percentage_error = abs_error / actual[i]
        sum_of_percentage_error += percentage_error
        MAPE = sum_of_percentage_error / (i + 1) * 100
        mape_list.append(float(MAPE))
    return mape_list

def fitting(closing_prices, stock_symbol):
    logging.debug(f'fitting called with closing_prices={closing_prices}, stock_symbol={stock_symbol}')
    Fitting_S_n_list = []
    v_list = []
    first_run = True
    
    if len(closing_prices) < 4:
        st.error("Tidak cukup data untuk melakukan fitting. Minimal 4 data point diperlukan.")
        return [], []
    
    for i in range(3):
        Fitting_S_n_list.append(float(closing_prices[i]))

    for i in range(3, len(closing_prices)):
        S_minus_1 = closing_prices[i - 3]
        S_0 = closing_prices[i - 2]
        S_1 = closing_prices[i - 1]
        S_2 = closing_prices[i]
        
        v_0 = determine_v_n(S_0, S_minus_1)
        v_1 = determine_v_n(S_1, S_0)
        v_2 = determine_v_n(S_2, S_1)
        
        if first_run:
            v_list.append(v_0)
            v_list.append(v_1)
            first_run = False
        v_list.append(v_2)

        try:
            alpha_n = determine_alpha_n(S_minus_1,S_0, S_1, S_2)
            beta_n = determine_beta_n(S_minus_1,S_1, S_2, alpha_n)
            h_n = determine_h_n(v_0, alpha_n, beta_n)
            condition_1 = (v_2 + (beta_n / alpha_n)) * v_2
            S_n = determine_s_n(S_minus_1, alpha_n, beta_n, h_n, condition_1, S_2, v_2, v_0)
        except (ZeroDivisionError) as e:
            logging.warning(f"Error in calculation at index {i}: {e}. Using fallback.")
            S_n = S_2

        Fitting_S_n_list.append(float(S_n))
        logging.debug(f'Appended S_n={S_n} to Fitting_S_n_list')
    
    return Fitting_S_n_list, v_list

def forecasting(Fitting_S_n_list, forecast_data, stock_symbol):
    """
    Updated forecasting function with proper date alignment
    """
    if len(Fitting_S_n_list) < 4:
        st.error("Tidak cukup data fitting untuk melakukan forecasting.")
        return [], [], []
    
    if forecast_data is None or forecast_data.empty:
        st.warning("Tidak ada data forecast yang tersedia.")
        return [], [], []
        
    # Get actual forecast prices and dates
    actual_forecast_prices = forecast_data['Close'].tolist()
    forecast_dates = forecast_data.index.tolist()
    
    # Initialize with last 4 fitting values
    fitting_S_last = Fitting_S_n_list[-4:].copy()
    
    forecast_days = len(actual_forecast_prices)
    
    if forecast_days <= 0:
        st.warning("Tidak terdapat data untuk melakukan forecast.")
        return [], [], []

    # Perform forecasting
    S_forecast_list = []
    
    for i in range(forecast_days):
        # Use the last 4 values for calculation
        if len(fitting_S_last) >= 4:
            S_minus_1 = fitting_S_last[-4]
            S_0 = fitting_S_last[-3]
            S_1 = fitting_S_last[-2]
            S_2 = fitting_S_last[-1]
            
            v_0 = determine_v_n(S_0, S_minus_1)
            v_2 = determine_v_n(S_2, S_1)
            
            try:
                alpha_n = determine_alpha_n(S_minus_1, S_0, S_1, S_2)
                beta_n = determine_beta_n(S_0, S_1, S_2, alpha_n)
                h_n = determine_h_n(v_0, alpha_n, beta_n)
                condition_1 = (v_2 + (beta_n / alpha_n)) * v_2
                S_n = determine_s_n(S_minus_1, alpha_n, beta_n, h_n, condition_1, S_2, v_2, v_0)
            except (ZeroDivisionError, Exception) as e:
                logging.warning(f"Error in forecast at step {i}: {e}. Using previous value.")
                S_n = S_2
                
            S_forecast_list.append(float(S_n))
            fitting_S_last.append(float(S_n))
        else:
            # Fallback if not enough data
            S_forecast_list.append(fitting_S_last[-1] if fitting_S_last else 0)
    
    logging.info(f"Generated {len(S_forecast_list)} forecast points")
    
    return S_forecast_list, forecast_dates, actual_forecast_prices