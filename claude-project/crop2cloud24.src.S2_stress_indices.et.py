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
import pyet
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration
DB_PATH = 'mpc_data.db'
ELEVATION = 876  # meters
LATITUDE = 41.15  # degrees
LONGITUDE = -100.77  # degrees
WIND_HEIGHT = 3  # meters

class CustomFormatter(logging.Formatter):
    def format(self, record):
        return f"{datetime.now(pytz.timezone('America/Chicago')).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} CST - {record.levelname} - {record.message}"

logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(CustomFormatter())
logger.addHandler(handler)

def get_db_connection():
    return sqlite3.connect(DB_PATH)

class ETCalculator:
    def __init__(self, db_path, plot_numbers):
        self.db_path = db_path
        self.plot_numbers = plot_numbers
        self.plot_tables = [f"LINEAR_CORN_trt1_plot_{num}" for num in plot_numbers]

    def reset_et_columns(self):
        logger.info("Resetting ET columns to NULL for all plots")
        conn = get_db_connection()
        cursor = conn.cursor()

        for table in self.plot_tables:
            logger.info(f"Resetting ET column in {table}")
            cursor.execute(f"UPDATE `{table}` SET et = NULL")
            rows_affected = cursor.rowcount
            logger.info(f"Reset {rows_affected} rows in {table}")
            conn.commit()

        conn.close()
        logger.info("ET columns reset completed")

    def get_weather_data(self):
        conn = get_db_connection()
        query = """
        SELECT TIMESTAMP, Ta_2m_Avg, TaMax_2m, TaMin_2m, RH_2m_Avg, 
               RHMax_2m, RHMin_2m, WndAveSpd_3m, Solar_2m_Avg
        FROM weather_data
        ORDER BY TIMESTAMP
        """
        df = pd.read_sql_query(query, conn, parse_dates=['TIMESTAMP'])
        df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], utc=True)
        conn.close()
        logger.info(f"Weather data retrieved. Shape: {df.shape}")
        logger.info(f"Sample weather data timestamps:\n{df['TIMESTAMP'].head().to_string()}")
        return df

    def calculate_et(self, df):
        logger.info(f"Input dataframe shape: {df.shape}")
        logger.info(f"Input dataframe dtypes:\n{df.dtypes}")
        logger.info(f"Input dataframe head:\n{df.head()}")
        
        # Resample to daily, handling missing data
        df_daily = df.set_index('TIMESTAMP').resample('D').mean()
        logger.info(f"Shape of daily data before dropna: {df_daily.shape}")
        logger.info(f"Columns with all NaN:\n{df_daily.columns[df_daily.isna().all()].tolist()}")
        
        df_daily = df_daily.dropna(subset=['Ta_2m_Avg', 'TaMax_2m', 'TaMin_2m', 'RHMax_2m', 'RHMin_2m', 'WndAveSpd_3m', 'Solar_2m_Avg'])
        logger.info(f"Shape of daily data after dropna: {df_daily.shape}")
        
        if df_daily.empty:
            logger.warning("Warning: Dataframe is empty after resampling and dropping NaN values.")
            return pd.DataFrame(columns=['TIMESTAMP', 'et'])
        
        # Convert solar radiation to MJ/m^2/day
        df_daily['Solar_2m_Avg_MJ'] = df_daily['Solar_2m_Avg'] * 0.0864
        
        lat_rad = LATITUDE * np.pi / 180
        # Prepare input data, checking for missing values
        inputs = {
            'tmean': df_daily['Ta_2m_Avg'],
            'wind': df_daily['WndAveSpd_3m'],
            'rs': df_daily['Solar_2m_Avg_MJ'],
            'tmax': df_daily['TaMax_2m'],
            'tmin': df_daily['TaMin_2m'],
            'rh': (df_daily['RHMax_2m'] + df_daily['RHMin_2m']) / 2,
            'elevation': ELEVATION,
            'lat': lat_rad
        }
        
        # Log the shape of each input
        for key, value in inputs.items():
            if isinstance(value, (pd.Series, np.ndarray)):
                logger.info(f"Shape of {key}: {value.shape}")
            else:
                logger.info(f"Value of {key}: {value}")
        
        # Calculate ET
        df_daily['et'] = pyet.combination.pm_asce(**inputs)
        
        # Reset index to get TIMESTAMP as a column and return only TIMESTAMP and et
        return df_daily.reset_index()[['TIMESTAMP', 'et']]

    def update_et_in_plot_tables(self, df_et):
        logger.info("Updating ET values in plot tables")
        
        conn = get_db_connection()
        cursor = conn.cursor()

        for table in self.plot_tables:
            logger.info(f"Processing table: {table}")
            
            # Update ET values
            rows_updated = 0
            for _, row in df_et.iterrows():
                et_timestamp = row['TIMESTAMP']
                # Use a 1-hour window to match timestamps
                cursor.execute(f"""
                UPDATE `{table}`
                SET et = ?
                WHERE TIMESTAMP BETWEEN ? AND ?
                """, (row['et'], 
                    (et_timestamp - timedelta(minutes=30)).strftime('%Y-%m-%d %H:%M:%S'),
                    (et_timestamp + timedelta(minutes=30)).strftime('%Y-%m-%d %H:%M:%S')))
                rows_updated += cursor.rowcount
                
                # Immediately verify the update
                cursor.execute(f"""
                SELECT TIMESTAMP, et FROM `{table}`
                WHERE TIMESTAMP BETWEEN ? AND ?
                """, ((et_timestamp - timedelta(minutes=30)).strftime('%Y-%m-%d %H:%M:%S'),
                      (et_timestamp + timedelta(minutes=30)).strftime('%Y-%m-%d %H:%M:%S')))
                updated_data = cursor.fetchone()
                if updated_data:
                    logger.info(f"Updated ET data in {table}: Timestamp={updated_data[0]}, ET={updated_data[1]}")
                else:
                    logger.warning(f"No ET data found immediately after update for timestamp: {et_timestamp}")
            
            conn.commit()
            logger.info(f"Updated {rows_updated} rows in {table}")
            
            # Log new ET values
            cursor.execute(f"SELECT MIN(et), MAX(et), AVG(et) FROM `{table}` WHERE et IS NOT NULL")
            min_et, max_et, avg_et = cursor.fetchone()
            logger.info(f"New ET values in {table}: Min: {min_et}, Max: {max_et}, Avg: {avg_et}")
            
            # Sample of updated data
            cursor.execute(f"SELECT TIMESTAMP, et FROM `{table}` WHERE et IS NOT NULL LIMIT 5")
            sample_data = cursor.fetchall()
            logger.info(f"Sample of updated ET data in {table}: {sample_data}")
            
        conn.close()
        logger.info(f"Successfully updated ET values in all plot tables.")

def compute_et(plot_numbers):
    start_time = time.time()
    logger.info(f"Starting ET computation for plots: {plot_numbers}")
    
    et_calculator = ETCalculator(DB_PATH, plot_numbers)
    
    # Reset ET columns to NULL
    et_calculator.reset_et_columns()
    
    df = et_calculator.get_weather_data()
    
    if df.empty:
        logger.info("No weather data available")
        return None
    
    logger.info(f"Processing {len(df)} rows of weather data")
    
    # Compute daily ET
    df_et = et_calculator.calculate_et(df)
    
    if df_et.empty:
        logger.info("No ET values computed")
        return None
    
    logger.info(f"ET computation results: Min: {df_et['et'].min():.2f}, Max: {df_et['et'].max():.2f}, Avg: {df_et['et'].mean():.2f}")
    logger.info(f"Sample of computed ET values:\n{df_et.head().to_string()}")
    
    et_calculator.update_et_in_plot_tables(df_et)
    
    end_time = time.time()
    duration = end_time - start_time
    logger.info(f"ET computation completed. Rows processed: {len(df_et)}")
    logger.info(f"Total execution time: {duration:.2f} seconds")
    return f"ET computation completed for plots {plot_numbers}. Rows processed: {len(df_et)}. Execution time: {duration:.2f} seconds"

def main():
    plot_numbers = [5001, 5003, 5012, 5020, 5006, 5001, 5018, 5025, 5007, 5009, 5027]
    result = compute_et(plot_numbers)
    print(result)

if __name__ == "__main__":
    main()