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
CST = pytz.timezone('America/Chicago')

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

def get_plot_data(conn, plot_number, irt_column):
    if DAYS_BACK is None:
        query = f"""
        SELECT TIMESTAMP, {irt_column}, Ta_2m_Avg
        FROM plot_{plot_number}
        ORDER BY TIMESTAMP
        """
    else:
        query = f"""
        SELECT TIMESTAMP, {irt_column}, Ta_2m_Avg
        FROM plot_{plot_number}
        WHERE TIMESTAMP >= datetime('now', '-{DAYS_BACK} days')
        ORDER BY TIMESTAMP
        """
    df = pd.read_sql_query(query, conn, parse_dates=['TIMESTAMP'])
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], utc=True).dt.tz_convert(CST)
    return df

def update_cwsi_th2(conn, plot_number, df_cwsi):
    logger.info(f"Updating CWSI-TH2 for plot {plot_number}")

    cursor = conn.cursor()

    # Check if the column exists, if not, create it
    cursor.execute(f"PRAGMA table_info(plot_{plot_number})")
    columns = [column[1] for column in cursor.fetchall()]
    if 'cwsi-th2' not in columns:
        cursor.execute(f"ALTER TABLE plot_{plot_number} ADD COLUMN 'cwsi-th2' REAL")
        conn.commit()
        logger.info(f"Added 'cwsi-th2' column to plot_{plot_number} table")

    rows_updated = 0
    for _, row in df_cwsi.iterrows():
        timestamp = row['TIMESTAMP'].tz_convert('UTC')
        start_time = timestamp - timedelta(minutes=30)
        end_time = timestamp + timedelta(minutes=30)
        
        cursor.execute(f"""
        UPDATE plot_{plot_number}
        SET 'cwsi-th2' = ?
        WHERE TIMESTAMP BETWEEN ? AND ?
        """, (row['cwsi-th2'], start_time.strftime('%Y-%m-%d %H:%M:%S'), end_time.strftime('%Y-%m-%d %H:%M:%S')))
        
        if cursor.rowcount > 0:
            rows_updated += cursor.rowcount
        else:
            logger.warning(f"No matching row found for timestamp: {timestamp}")

    conn.commit()
    
    logger.info(f"Successfully updated CWSI-TH2 for plot {plot_number}. Rows updated: {rows_updated}")
    
    # Check for any rows that weren't updated
    cursor.execute(f"""
    SELECT COUNT(*) FROM plot_{plot_number}
    WHERE 'cwsi-th2' IS NULL
    """)
    unupdated_rows = cursor.fetchone()[0]
    logger.info(f"Rows not updated: {unupdated_rows}")

    return rows_updated

def calculate_cwsi_th2(df, irt_column):
    # Calculate Tc - Ta
    df['Tc_Ta'] = df[irt_column] - df['Ta_2m_Avg']
    
    # Calculate lower and upper baselines
    Tc_Ta_l = df['Tc_Ta'].quantile(0.05)  # Lower baseline (non-water stressed)
    Tc_Ta_u = df['Tc_Ta'].quantile(0.95)  # Upper baseline (water stressed)
    
    # Calculate CWSI-TH2
    df['cwsi-th2'] = (df['Tc_Ta'] - Tc_Ta_l) / (Tc_Ta_u - Tc_Ta_l)
    
    # Clip CWSI-TH2 values to be between 0 and 1
    df['cwsi-th2'] = df['cwsi-th2'].clip(0, 1)
    
    return df

def compute_cwsi(plot_number):
    start_time = time.time()
    logger.info(f"Starting CWSI-TH2 computation for plot {plot_number}")
    
    conn = get_db_connection()

    irt_column = f'IRT{plot_number}B1xx24' if plot_number == '5006' else f'IRT{plot_number}C1xx24' if plot_number == '5010' else f'IRT{plot_number}A1xx24'
    df = get_plot_data(conn, plot_number, irt_column)
    
    if df.empty:
        logger.info(f"No data for plot {plot_number}")
        conn.close()
        return None
    
    logger.info(f"Processing {len(df)} rows for plot {plot_number}")
    
    # Filter for 12 PM to 4 PM CST
    df = df[(df['TIMESTAMP'].dt.hour >= 12) & (df['TIMESTAMP'].dt.hour < 16)]
    
    if df.empty:
        logger.info(f"No data within 12 PM to 4 PM CST for plot {plot_number}")
        conn.close()
        return None
    
    logger.info(f"Calculating CWSI-TH2 for {len(df)} rows")
    df_with_cwsi = calculate_cwsi_th2(df, irt_column)
    
    df_cwsi = df_with_cwsi[['TIMESTAMP', 'cwsi-th2']].dropna()
    
    df_cwsi['TIMESTAMP'] = df_cwsi['TIMESTAMP'].dt.tz_convert('UTC')
    rows_updated = update_cwsi_th2(conn, plot_number, df_cwsi)
    
    conn.close()
    
    end_time = time.time()
    duration = end_time - start_time
    logger.info(f"CWSI-TH2 computation completed for plot {plot_number}.")
    logger.info(f"Rows processed: {len(df_cwsi)}, Rows updated in database: {rows_updated}")
    logger.info(f"Total execution time: {duration:.2f} seconds")
    return f"CWSI-TH2 computation completed for plot {plot_number}. Rows processed: {len(df_cwsi)}, Rows updated: {rows_updated}. Execution time: {duration:.2f} seconds"

def main():
    plot_numbers = ['5006', '5010', '5023']
    for plot_number in plot_numbers:
        result = compute_cwsi(plot_number)
        print(result)

if __name__ == "__main__":
    main()