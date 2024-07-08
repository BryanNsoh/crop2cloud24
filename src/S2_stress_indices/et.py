import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import logging
import sys
import time
import refet

# Configuration
DB_PATH = 'mpc_data.db'
ELEVATION = 876  # meters
LATITUDE = 41.15  # degrees
LONGITUDE = -100.77  # degrees
WIND_HEIGHT = 3  # meters

class CustomFormatter(logging.Formatter):
    def format(self, record):
        return f"{datetime.now(pytz.timezone('America/Chicago')).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} CST - {record.levelname} - {record.msg}"

logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(CustomFormatter())
logger.addHandler(handler)

def get_db_connection():
    return sqlite3.connect(DB_PATH)

def get_weather_data(conn):
    query = """
    SELECT TIMESTAMP, Ta_2m_Avg, TaMax_2m, TaMin_2m, RH_2m_Avg, 
           Dp_2m_Avg, WndAveSpd_3m, Solar_2m_Avg
    FROM weather_data
    ORDER BY TIMESTAMP
    """
    df = pd.read_sql_query(query, conn, parse_dates=['TIMESTAMP'])
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], utc=True)
    logger.info(f"Weather data retrieved. Shape: {df.shape}")
    logger.info(f"Sample weather data timestamps:\n{df['TIMESTAMP'].head().to_string()}")
    return df

def compute_daily_et(df):
    et_values = []
    for _, row in df.iterrows():
        try:
            ea = refet.calcs._sat_vapor_pressure(row['Dp_2m_Avg'])
            rs = row['Solar_2m_Avg'] * 0.0864
            
            et = refet.Daily(
                tmin=row['TaMin_2m'],
                tmax=row['TaMax_2m'],
                ea=ea,
                rs=rs,
                uz=row['WndAveSpd_3m'],
                zw=WIND_HEIGHT,
                elev=ELEVATION,
                lat=LATITUDE,
                doy=row['TIMESTAMP'].timetuple().tm_yday,
                method='asce'
            ).etr()
            et_values.append(float(et.item()))
        except Exception as e:
            logger.error(f"Error computing ET for date {row['TIMESTAMP']}: {str(e)}")
            et_values.append(None)
    
    return et_values

def update_et_in_plot_tables(conn, df_et):
    logger.info("Updating ET values in plot tables")
    
    plot_tables = ['plot_5006', 'plot_5010', 'plot_5023']
    
    cursor = conn.cursor()
    for table in plot_tables:
        # Log current ET values
        cursor.execute(f"SELECT MIN(et), MAX(et), AVG(et) FROM {table}")
        min_et, max_et, avg_et = cursor.fetchone()
        logger.info(f"Current ET values in {table}: Min: {min_et}, Max: {max_et}, Avg: {avg_et}")
        
        # Update ET values
        rows_updated = 0
        for _, row in df_et.iterrows():
            et_timestamp = row['TIMESTAMP']
            # Use a 1-hour window to match timestamps
            cursor.execute(f"""
            UPDATE {table}
            SET et = ?, is_actual = 1
            WHERE TIMESTAMP BETWEEN ? AND ?
            """, (row['et'], 
                  (et_timestamp - timedelta(minutes=30)).strftime('%Y-%m-%d %H:%M:%S'),
                  (et_timestamp + timedelta(minutes=30)).strftime('%Y-%m-%d %H:%M:%S')))
            rows_updated += cursor.rowcount
        
        conn.commit()
        logger.info(f"Updated {rows_updated} rows in {table}")
        
        # Log new ET values
        cursor.execute(f"SELECT MIN(et), MAX(et), AVG(et) FROM {table}")
        min_et, max_et, avg_et = cursor.fetchone()
        logger.info(f"New ET values in {table}: Min: {min_et}, Max: {max_et}, Avg: {avg_et}")
    
    logger.info(f"Successfully updated ET values in all plot tables.")

def compute_et():
    start_time = time.time()
    logger.info("Starting ET computation")
    
    conn = get_db_connection()
    df = get_weather_data(conn)
    
    if df.empty:
        logger.info("No weather data available")
        conn.close()
        return None
    
    logger.info(f"Processing {len(df)} rows of weather data")
    
    # Compute daily ET
    df['et'] = compute_daily_et(df)
    df_et = df[['TIMESTAMP', 'et']].dropna()
    
    logger.info(f"ET computation results: Min: {df_et['et'].min():.2f}, Max: {df_et['et'].max():.2f}, Avg: {df_et['et'].mean():.2f}")
    logger.info(f"Sample of computed ET values:\n{df_et.head().to_string()}")
    
    update_et_in_plot_tables(conn, df_et)
    
    conn.close()
    
    end_time = time.time()
    duration = end_time - start_time
    logger.info(f"ET computation completed. Rows processed: {len(df_et)}")
    logger.info(f"Total execution time: {duration:.2f} seconds")
    return f"ET computation completed. Rows processed: {len(df_et)}. Execution time: {duration:.2f} seconds"

def main():
    result = compute_et()
    print(result)

if __name__ == "__main__":
    main()