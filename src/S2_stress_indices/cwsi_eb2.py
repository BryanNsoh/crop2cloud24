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
CST = pytz.timezone('America/Chicago')
REFERENCE_HOUR = 14  # 2 PM CST
REFERENCE_DATE = datetime(2024, 7, 2, REFERENCE_HOUR, 0, 0, tzinfo=CST)  # July 2, 2024 at 2 PM CST

class CustomFormatter(logging.Formatter):
    def format(self, record):
        record.message = record.getMessage()
        return f"{datetime.now(CST).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} CST - {record.levelname} - {record.message}"

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
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], utc=True).dt.tz_convert(CST)
    df['Date'] = df['TIMESTAMP'].dt.date
    daily_rain = df.groupby('Date')['Rain_1m_Tot'].sum()
    reference_date = daily_rain[daily_rain >= RAINFALL_THRESHOLD].index[-1]
    return pd.Timestamp(reference_date, tz=CST)

def get_plot_data(conn, plot_number, irt_column):
    if DAYS_BACK is None:
        query = f"""
        SELECT TIMESTAMP, {irt_column}
        FROM plot_{plot_number}
        ORDER BY TIMESTAMP
        """
    else:
        query = f"""
        SELECT TIMESTAMP, {irt_column}
        FROM plot_{plot_number}
        WHERE TIMESTAMP >= datetime('now', '-{DAYS_BACK} days')
        ORDER BY TIMESTAMP
        """
    df = pd.read_sql_query(query, conn, parse_dates=['TIMESTAMP'])
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], utc=True).dt.tz_convert(CST)
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
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], utc=True).dt.tz_convert(CST)
    return df

def update_cwsi_eb2(conn, plot_number, df_cwsi):
    logger.info(f"Updating CWSI-EB2 for plot {plot_number}")

    cursor = conn.cursor()

    # Check if the column exists, if not, create it
    cursor.execute(f"PRAGMA table_info(plot_{plot_number})")
    columns = [column[1] for column in cursor.fetchall()]
    if 'cwsi-eb2' not in columns:
        cursor.execute(f"ALTER TABLE plot_{plot_number} ADD COLUMN 'cwsi-eb2' REAL")
        conn.commit()
        logger.info(f"Added 'cwsi-eb2' column to plot_{plot_number} table")

    rows_updated = 0
    for _, row in df_cwsi.iterrows():
        timestamp = row['TIMESTAMP'].tz_convert('UTC')
        start_time = timestamp - timedelta(minutes=30)
        end_time = timestamp + timedelta(minutes=30)
        
        cursor.execute(f"""
        UPDATE plot_{plot_number}
        SET 'cwsi-eb2' = ?, is_actual = 1
        WHERE TIMESTAMP BETWEEN ? AND ?
        """, (row['cwsi-eb2'], start_time.strftime('%Y-%m-%d %H:%M:%S'), end_time.strftime('%Y-%m-%d %H:%M:%S')))
        
        if cursor.rowcount > 0:
            rows_updated += cursor.rowcount
        else:
            logger.warning(f"No matching row found for timestamp: {timestamp}")

    conn.commit()
    
    logger.info(f"Successfully updated CWSI-EB2 for plot {plot_number}. Rows updated: {rows_updated}")
    
    # Check for any rows that weren't updated
    cursor.execute(f"""
    SELECT COUNT(*) FROM plot_{plot_number}
    WHERE 'cwsi-eb2' IS NULL OR is_actual != 1
    """)
    unupdated_rows = cursor.fetchone()[0]
    logger.info(f"Rows not updated: {unupdated_rows}")

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
    
    # Filter data for the reference day within the specified hour
    reference_time_start = reference_date.replace(minute=0, second=0, microsecond=0)
    reference_time_end = reference_date.replace(minute=59, second=59, microsecond=999999)
    reference_data = df[(df['TIMESTAMP'] >= reference_time_start) & (df['TIMESTAMP'] <= reference_time_end)]
    
    if reference_data.empty:
        logger.warning(f"No reference data available for date: {reference_date}")
        return None

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
    
    # Calculate CWSI-EB2 (Eq. 2a)
    df['cwsi-eb2'] = (df['Tc_Ta'] - df['Tc_Ta_lower']) / (df['Tc_Ta_upper'] - df['Tc_Ta_lower'])
    
    # Exclude values outside the 0 to 1 range
    df['cwsi-eb2'] = df['cwsi-eb2'].clip(0, 1)
    
    return df

def compute_cwsi(plot_number):
    start_time = time.time()
    logger.info(f"Starting CWSI-EB2 computation for plot {plot_number}")
    
    conn = get_db_connection()
    
    reference_date = REFERENCE_DATE
    logger.info(f"Using reference date: {reference_date}")

    # Get rainfall data for reference date and following day
    rain_query = f"""
    SELECT TIMESTAMP, Rain_1m_Tot
    FROM mesonet_data
    WHERE DATE(TIMESTAMP) IN ('{reference_date.date()}', '{(reference_date + timedelta(days=1)).date()}')
    """
    rain_df = pd.read_sql_query(rain_query, conn, parse_dates=['TIMESTAMP'])
    rain_df['TIMESTAMP'] = pd.to_datetime(rain_df['TIMESTAMP'], utc=True).dt.tz_convert(CST)
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
    df = df.set_index('TIMESTAMP').resample('h').mean().reset_index()
    
    # Filter for 12 PM to 5 PM CST
    df = df[(df['TIMESTAMP'].dt.hour >= 12) & (df['TIMESTAMP'].dt.hour < 17)]
    
    if df.empty:
        logger.info(f"No data within 12 PM to 5 PM CST for plot {plot_number}")
        conn.close()
        return None
    
    start_time_weather = df['TIMESTAMP'].min()
    end_time_weather = df['TIMESTAMP'].max()
    
    weather_data = get_weather_data(conn, start_time_weather, end_time_weather)
    
    df = df.sort_values('TIMESTAMP')
    weather_data = weather_data.sort_values('TIMESTAMP')
    
    df = pd.merge_asof(df, weather_data, on='TIMESTAMP', direction='nearest')
    
    logger.info(f"Calculating CWSI-EB2 for {len(df)} rows")
    df_with_cwsi = calculate_cwsi_eb2(df, irt_column, reference_date)
    
    if df_with_cwsi is None:
        logger.warning(f"Skipping CWSI-EB2 computation for plot {plot_number} due to missing reference data")
        conn.close()
        return None
    
    df_cwsi = df_with_cwsi[['TIMESTAMP', 'cwsi-eb2']].dropna()
    
    # Get reference day data within the specified hour
    reference_data = df_with_cwsi[(df_with_cwsi['TIMESTAMP'] >= reference_date.replace(minute=0, second=0, microsecond=0)) &
                                  (df_with_cwsi['TIMESTAMP'] <= reference_date.replace(minute=59, second=59, microsecond=999999))]
    reference_canopy_temp = reference_data[irt_column].iloc[0]
    reference_air_temp = reference_data['Ta_2m_Avg'].iloc[0]

    # Log detailed statistics
    logger.info(f"Reference date: {reference_date}")
    logger.info(f"Rainfall on reference date: {rain_reference_day:.2f} inches")
    logger.info(f"Rainfall on following day: {rain_following_day:.2f} inches")
    logger.info(f"Canopy temperature at {reference_date.strftime('%I:%M %p')} CST on reference date: {reference_canopy_temp:.2f}°C")
    logger.info(f"Air temperature at {reference_date.strftime('%I:%M %p')} CST on reference date: {reference_air_temp:.2f}°C")
    logger.info(f"CWSI-EB2 statistics:")
    logger.info(f"  Max CWSI-EB2: {df_cwsi['cwsi-eb2'].max():.4f}")
    logger.info(f"  Min CWSI-EB2: {df_cwsi['cwsi-eb2'].min():.4f}")
    logger.info(f"  Median CWSI-EB2: {df_cwsi['cwsi-eb2'].median():.4f}")
    logger.info(f"  CWSI-EB2 values out of range (< 0 or > 1): {((df_cwsi['cwsi-eb2'] < 0) | (df_cwsi['cwsi-eb2'] > 1)).sum()}")

    df_cwsi['TIMESTAMP'] = df_cwsi['TIMESTAMP'].dt.tz_convert('UTC')
    rows_updated = update_cwsi_eb2(conn, plot_number, df_cwsi)
    
    conn.close()
    
    end_time = time.time()
    duration = end_time - start_time
    logger.info(f"CWSI-EB2 computation completed for plot {plot_number}.")
    logger.info(f"Rows processed: {len(df_cwsi)}, Rows updated in database: {rows_updated}")
    logger.info(f"Total execution time: {duration:.2f} seconds")
    return f"CWSI-EB2 computation completed for plot {plot_number}. Rows processed: {len(df_cwsi)}, Rows updated: {rows_updated}. Execution time: {duration:.2f} seconds"

def main():
    plot_numbers = ['5006', '5010', '5023']
    for plot_number in plot_numbers:
        result = compute_cwsi(plot_number)
        print(result)

if __name__ == "__main__":
    main()