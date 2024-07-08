import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import logging
import time
from dotenv import load_dotenv
from crop2cloud24.src.utils import generate_plots

# Load environment variables from .env file
load_dotenv()

# Configuration
DAYS_BACK = None  # Set to None for all available data, or specify a number of days
DB_PATH = 'mpc_data.db'
RAINFALL_THRESHOLD = 0.6  # inches

class CustomFormatter(logging.Formatter):
    def format(self, record):
        record.message = record.getMessage()
        return f"{datetime.now(pytz.timezone('America/Chicago')).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} CST - {record.levelname} - {record.message}"

logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(CustomFormatter())
logger.addHandler(handler)

def get_db_connection():
    return sqlite3.connect(DB_PATH)

def get_reference_date(conn):
    query = """
    SELECT TIMESTAMP, Rain_1m_Tot
    FROM mesonet_data
    ORDER BY TIMESTAMP DESC
    """
    df = pd.read_sql_query(query, conn, parse_dates=['TIMESTAMP'])
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], utc=True)
    df['Date'] = df['TIMESTAMP'].dt.date
    daily_rain = df.groupby('Date')['Rain_1m_Tot'].sum()
    reference_date = daily_rain[daily_rain >= RAINFALL_THRESHOLD].index[-1]
    return pd.Timestamp(reference_date).tz_localize('UTC') + timedelta(days=1)

def get_plot_data(conn, plot_number, irt_column):
    if DAYS_BACK is None:
        query = f"""
        SELECT TIMESTAMP, {irt_column}, is_actual, cwsi
        FROM plot_{plot_number}
        ORDER BY TIMESTAMP
        """
    else:
        query = f"""
        SELECT TIMESTAMP, {irt_column}, is_actual, cwsi
        FROM plot_{plot_number}
        WHERE TIMESTAMP >= datetime('now', '-{DAYS_BACK} days')
        ORDER BY TIMESTAMP
        """
    df = pd.read_sql_query(query, conn, parse_dates=['TIMESTAMP'])
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], utc=True)
    return df

def get_weather_data(conn, start_time, end_time):
    query = """
    SELECT *
    FROM weather_data
    WHERE TIMESTAMP BETWEEN ? AND ?
    ORDER BY TIMESTAMP
    """
    start_time_str = start_time.strftime('%Y-%m-%d %H:%M:%S')
    end_time_str = end_time.strftime('%Y-%m-%d %H:%M:%S')
    df = pd.read_sql_query(query, conn, params=(start_time_str, end_time_str), parse_dates=['TIMESTAMP'])
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], utc=True)
    return df

def update_cwsi(conn, plot_number, df_cwsi):
    logger.info(f"Updating CWSI for plot {plot_number}")
    
    cursor = conn.cursor()
    rows_updated = 0
    for _, row in df_cwsi.iterrows():
        cursor.execute(f"""
        UPDATE plot_{plot_number}
        SET cwsi = ?, is_actual = 1
        WHERE TIMESTAMP = ?
        """, (row['cwsi'], row['TIMESTAMP'].strftime('%Y-%m-%d %H:%M:%S')))
        rows_updated += cursor.rowcount
    conn.commit()
    
    logger.info(f"Successfully updated CWSI for plot {plot_number}. Rows updated: {rows_updated}")
    return rows_updated

def saturated_vapor_pressure(T):
    return 0.6108 * np.exp(17.27 * T / (T + 237.3))

def vapor_pressure_deficit(T, RH):
    es = saturated_vapor_pressure(T)
    ea = es * (RH / 100)
    return es - ea

def calculate_cwsi_eb2(df, canopy_temp_column, reference_date):
    # Calculate VPD
    df['VPD'] = vapor_pressure_deficit(df['Ta_2m_Avg'], df['RH_2m_Avg'])
    
    # Calculate Tc - Ta
    df['Tc_Ta'] = df[canopy_temp_column] - df['Ta_2m_Avg']
    
    # Filter data for the reference day at 12 PM UTC (7 AM CST)
    reference_time = reference_date.replace(hour=12, minute=0, second=0, microsecond=0)
    reference_data = df[df['TIMESTAMP'] == reference_time]
    
    if reference_data.empty:
        logger.warning(f"No reference data available for date: {reference_date}")
        return df

    # Develop lower baseline (non-water stressed baseline)
    reference_tc_ta = reference_data['Tc_Ta'].iloc[0]
    reference_vpd = reference_data['VPD'].iloc[0]
    slope = reference_tc_ta / reference_vpd
    intercept = 0  # Assuming the line passes through the origin
    
    # Calculate lower baseline
    df['Tc_Ta_lower'] = slope * df['VPD'] + intercept
    
    # Calculate upper baseline using VPG method (Eq. 2d)
    df['VPG'] = saturated_vapor_pressure(df['Ta_2m_Avg'] + intercept) - saturated_vapor_pressure(df['Ta_2m_Avg'])
    df['Tc_Ta_upper'] = intercept + slope * df['VPG']
    
    # Calculate CWSI (Eq. 2a)
    df['cwsi'] = (df['Tc_Ta'] - df['Tc_Ta_lower']) / (df['Tc_Ta_upper'] - df['Tc_Ta_lower'])
    
    return df

def compute_cwsi(plot_number):
    start_time = time.time()
    logger.info(f"Starting CWSI computation for plot {plot_number}")
    
    conn = get_db_connection()
    
    reference_date = get_reference_date(conn)
    logger.info(f"Using reference date: {reference_date}")

    # Get rainfall data for reference date and following day
    rain_query = f"""
    SELECT TIMESTAMP, Rain_1m_Tot
    FROM mesonet_data
    WHERE DATE(TIMESTAMP) IN ('{reference_date.date()}', '{(reference_date + timedelta(days=1)).date()}')
    """
    rain_df = pd.read_sql_query(rain_query, conn, parse_dates=['TIMESTAMP'])
    rain_df['TIMESTAMP'] = pd.to_datetime(rain_df['TIMESTAMP'], utc=True)
    rain_reference_day = rain_df[rain_df['TIMESTAMP'].dt.date == reference_date.date()]['Rain_1m_Tot'].sum()
    rain_following_day = rain_df[rain_df['TIMESTAMP'].dt.date == (reference_date + timedelta(days=1)).date()]['Rain_1m_Tot'].sum()

    irt_column = f'IRT{plot_number}B1xx24' if plot_number == '5006' else f'IRT{plot_number}C1xx24' if plot_number == '5010' else f'IRT{plot_number}A1xx24'
    df = get_plot_data(conn, plot_number, irt_column)
    
    if df.empty:
        logger.info(f"No data for plot {plot_number}")
        conn.close()
        return None
    
    logger.info(f"Processing {len(df)} rows for plot {plot_number}")
    
    # Ensure we're working with hourly data
    df = df.set_index('TIMESTAMP').resample('H').mean().reset_index()
    
    # Filter for 12 PM to 5 PM UTC (7 AM to 12 PM CST)
    df = df[(df['TIMESTAMP'].dt.hour >= 12) & (df['TIMESTAMP'].dt.hour < 17)]
    
    if df.empty:
        logger.info(f"No data within 12 PM to 5 PM UTC for plot {plot_number}")
        conn.close()
        return None
    
    start_time_weather = df['TIMESTAMP'].min()
    end_time_weather = df['TIMESTAMP'].max()
    
    weather_data = get_weather_data(conn, start_time_weather, end_time_weather)
    
    df = df.sort_values('TIMESTAMP')
    weather_data = weather_data.sort_values('TIMESTAMP')
    
    df = pd.merge_asof(df, weather_data, on='TIMESTAMP', direction='nearest')
    
    logger.info(f"Calculating CWSI for {len(df)} rows")
    df_with_cwsi = calculate_cwsi_eb2(df, irt_column, reference_date)
    df_cwsi = df_with_cwsi[['TIMESTAMP', 'cwsi', 'is_actual']].dropna()
    
    # Get reference day data
    reference_data = df_with_cwsi[df_with_cwsi['TIMESTAMP'].dt.date == reference_date.date()]
    reference_canopy_temp = reference_data[reference_data['TIMESTAMP'].dt.hour == 12][irt_column].iloc[0]
    reference_air_temp = reference_data[reference_data['TIMESTAMP'].dt.hour == 12]['Ta_2m_Avg'].iloc[0]

    # Log detailed statistics
    logger.info(f"Reference date: {reference_date}")
    logger.info(f"Rainfall on reference date: {rain_reference_day:.2f} inches")
    logger.info(f"Rainfall on following day: {rain_following_day:.2f} inches")
    logger.info(f"Canopy temperature at 12 PM UTC (7 AM CST) on reference date: {reference_canopy_temp:.2f}°C")
    logger.info(f"Air temperature at 12 PM UTC (7 AM CST) on reference date: {reference_air_temp:.2f}°C")
    logger.info(f"CWSI statistics:")
    logger.info(f"  Max CWSI: {df_cwsi['cwsi'].max():.4f}")
    logger.info(f"  Min CWSI: {df_cwsi['cwsi'].min():.4f}")
    logger.info(f"  Median CWSI: {df_cwsi['cwsi'].median():.4f}")
    logger.info(f"  CWSI values out of range (< 0 or > 1): {((df_cwsi['cwsi'] < 0) | (df_cwsi['cwsi'] > 1)).sum()}")

    rows_updated = update_cwsi(conn, plot_number, df_cwsi)
    
    conn.close()
    
    end_time = time.time()
    duration = end_time - start_time
    logger.info(f"CWSI computation completed for plot {plot_number}.")
    logger.info(f"Rows processed: {len(df_cwsi)}, Rows updated in database: {rows_updated}")
    logger.info(f"Total execution time: {duration:.2f} seconds")
    return f"CWSI computation completed for plot {plot_number}. Rows processed: {len(df_cwsi)}, Rows updated: {rows_updated}. Execution time: {duration:.2f} seconds"

def main():
    plot_numbers = ['5006', '5010', '5023']
    for plot_number in plot_numbers:
        result = compute_cwsi(plot_number)
        print(result)
    
    # Generate plots using the imported function
    generate_plots(plot_numbers=plot_numbers, days=DAYS_BACK)

if __name__ == "__main__":
    main()