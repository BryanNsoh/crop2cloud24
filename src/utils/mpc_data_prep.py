# src/utils/mpc_data_prep.py

import os
import requests
import sqlite3
import pandas as pd
import numpy as np
from google.cloud import bigquery
from dotenv import load_dotenv
from datetime import datetime, timedelta
import pytz
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# BigQuery client
client = bigquery.Client()

# SQLite database
DB_NAME = 'mpc_data.db'

# Specify the number of historical days to retrieve
HISTORICAL_DAYS = 30

# Cloud function trigger URLs
TRIGGER_URLS = [
    'https://us-central1-crop2cloud24.cloudfunctions.net/compute-cwsi',
    'https://us-central1-crop2cloud24.cloudfunctions.net/compute-swsi',
    'https://us-central1-crop2cloud24.cloudfunctions.net/current-openweathermap',
    'https://us-central1-crop2cloud24.cloudfunctions.net/weather-updater',
    'https://us-central1-crop2cloud24.cloudfunctions.net/forecast_four_day_rolling',
]

def trigger_cloud_functions():
    for url in TRIGGER_URLS:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                logger.info(f"Successfully triggered: {url}")
            else:
                logger.warning(f"Failed to trigger: {url}. Status code: {response.status_code}")
        except requests.RequestException as e:
            logger.error(f"Error triggering {url}: {e}")

def get_data_with_history(table_name):
    end_time = datetime.now(pytz.UTC)
    start_time = end_time - timedelta(days=HISTORICAL_DAYS)
    
    query = f"""
    SELECT *
    FROM `crop2cloud24.weather.{table_name}`
    WHERE TIMESTAMP BETWEEN '{start_time}' AND '{end_time}'
    ORDER BY TIMESTAMP
    """
    
    logger.info(f"Executing query for {table_name}:\n{query}")
    df = client.query(query).to_dataframe()
    logger.info(f"Raw data retrieved for {table_name}. Shape: {df.shape}")
    
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], utc=True)
    df = df.dropna(subset=['TIMESTAMP'])
    
    logger.info(f"{table_name} data range: {df['TIMESTAMP'].min()} to {df['TIMESTAMP'].max()}")
    logger.info(f"{table_name} data shape: {df.shape}")
    logger.info(f"{table_name} columns: {df.columns.tolist()}")
    logger.info(f"Sample of {table_name} data:\n{df.head().to_string()}")
    
    return df

def resample_hourly(df, is_forecast=False):
    logger.info(f"Resampling data to hourly intervals. Initial shape: {df.shape}")
    df = df.set_index('TIMESTAMP')
    
    if is_forecast:
        # For forecast data, interpolate to hourly frequency
        df_resampled = df.resample('H').interpolate(method='time')
        logger.info("Interpolated forecast data to hourly frequency")
    else:
        # For non-forecast data, use the previous resampling method
        columns_to_average = [col for col in df.columns if col != 'Rain_1m_Tot']
        resampling_dict = {col: 'mean' for col in columns_to_average}
        if 'Rain_1m_Tot' in df.columns:
            resampling_dict['Rain_1m_Tot'] = 'sum'
        df_resampled = df.resample('H').agg(resampling_dict)
    
    # Reset index to make TIMESTAMP a column again
    df_resampled = df_resampled.reset_index()
    
    logger.info(f"Resampled data shape: {df_resampled.shape}")
    logger.info(f"Sample of resampled data:\n{df_resampled.head().to_string()}")
    
    return df_resampled

def get_treatment_1_plots():
    query = """
    SELECT table_name
    FROM `crop2cloud24.LINEAR_CORN_trt1.INFORMATION_SCHEMA.TABLES`
    WHERE table_name LIKE 'plot_%'
    """
    logger.info(f"Executing query to get treatment 1 plots:\n{query}")
    tables = [row.table_name for row in client.query(query).result()]
    plot_numbers = [int(table_name.split('_')[1]) for table_name in tables]
    logger.info(f"Found plot numbers: {plot_numbers}")
    return sorted(plot_numbers)

def get_plot_data(plot_number):
    end_time = datetime.now(pytz.UTC)
    start_time = end_time - timedelta(days=HISTORICAL_DAYS)
    
    query = f"""
    SELECT *
    FROM `crop2cloud24.LINEAR_CORN_trt1.plot_{plot_number}`
    WHERE TIMESTAMP BETWEEN '{start_time}' AND '{end_time}'
    ORDER BY TIMESTAMP
    """
    
    logger.info(f"Executing query for plot {plot_number}:\n{query}")
    df = client.query(query).to_dataframe()
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], utc=True)
    
    logger.info(f"Plot {plot_number} data range: {df['TIMESTAMP'].min()} to {df['TIMESTAMP'].max()}")
    logger.info(f"Plot {plot_number} data shape: {df.shape}")
    logger.info(f"Plot {plot_number} columns: {df.columns.tolist()}")
    logger.info(f"Sample of plot {plot_number} data:\n{df.head().to_string()}")
    
    return df

def create_sqlite_db():
    conn = sqlite3.connect(DB_NAME)
    conn.close()
    logger.info(f"Created SQLite database: {DB_NAME}")

def create_plot_table(plot_number, df):
    conn = sqlite3.connect(DB_NAME)
    table_name = f"plot_{plot_number}"
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    conn.close()
    logger.info(f"Created table: {table_name}")
    logger.info(f"Sample of data in {table_name}:\n{df.head().to_string()}")

def add_weather_and_forecast_data_to_tables(weather_df, forecast_df):
    logger.info(f"Adding weather and forecast data to plot tables. Weather data shape: {weather_df.shape}, Forecast data shape: {forecast_df.shape}")
    logger.info(f"Weather data columns: {weather_df.columns.tolist()}")
    logger.info(f"Forecast data columns: {forecast_df.columns.tolist()}")

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # Get list of plot tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'plot_%'")
    plot_tables = [row[0] for row in cursor.fetchall()]
    logger.info(f"Found plot tables: {plot_tables}")

    for table in plot_tables:
        logger.info(f"Processing table: {table}")
        
        # Read the plot data
        plot_df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
        plot_df['TIMESTAMP'] = pd.to_datetime(plot_df['TIMESTAMP'])
        
        # Merge weather data
        merged_df = pd.merge_asof(plot_df, weather_df, on='TIMESTAMP', direction='nearest', tolerance=pd.Timedelta('1h'))
        
        # Merge forecast data without aligning timestamps
        forecast_columns = [col for col in forecast_df.columns if col != 'TIMESTAMP']
        for col in forecast_columns:
            merged_df[f'{col}_rolling'] = np.nan

        merged_df = merged_df.set_index('TIMESTAMP')
        forecast_df = forecast_df.set_index('TIMESTAMP')
        
        merged_df.update(forecast_df, overwrite=False)
        merged_df = merged_df.reset_index()
        
        # Update the table
        merged_df.to_sql(table, conn, if_exists='replace', index=False)
        
        logger.info(f"Updated table {table}. New shape: {merged_df.shape}")
        logger.info(f"Sample of updated data in {table}:\n{merged_df.head().to_string()}")

    conn.close()
    logger.info("Added weather and forecast data to all plot tables")

def main():
    logger.info("Starting MPC data preparation")
    logger.info("Triggering cloud functions...")
    trigger_cloud_functions()

    logger.info("Retrieving and processing weather data...")
    try:
        mesonet_data = get_data_with_history('current-weather-mesonet')
        rolling_forecast = get_data_with_history('forecast_four_day_rolling')

        logger.info("Resampling data to hourly intervals...")
        mesonet_hourly = resample_hourly(mesonet_data)
        rolling_forecast_hourly = resample_hourly(rolling_forecast, is_forecast=True)

        logger.info("Creating SQLite database...")
        create_sqlite_db()

        logger.info("Retrieving treatment 1 plots...")
        plots = get_treatment_1_plots()
        logger.info(f"Found {len(plots)} plots: {plots}")

        for plot in plots:
            logger.info(f"Processing plot {plot}...")
            try:
                plot_data = get_plot_data(plot)
                plot_data_hourly = resample_hourly(plot_data)
                create_plot_table(plot, plot_data_hourly)
            except Exception as e:
                logger.error(f"Error processing plot {plot}: {str(e)}")
                continue

        logger.info("Adding weather and forecast data to plot tables...")
        add_weather_and_forecast_data_to_tables(mesonet_hourly, rolling_forecast_hourly)

        logger.info("Data preparation complete.")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()