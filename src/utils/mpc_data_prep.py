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
HISTORICAL_DAYS = 12

# Cloud function trigger URLs
TRIGGER_URLS = [
    # 'https://us-central1-crop2cloud24.cloudfunctions.net/compute-cwsi',
    # 'https://us-central1-crop2cloud24.cloudfunctions.net/compute-swsi',
    # 'https://us-central1-crop2cloud24.cloudfunctions.net/current-openweathermap',
    # 'https://us-central1-crop2cloud24.cloudfunctions.net/weather-updater',
    # 'https://us-central1-crop2cloud24.cloudfunctions.net/forecast_four_day_rolling',
    # 'https://us-central1-crop2cloud24.cloudfunctions.net/forecast_four_day_static'
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

def merge_duplicate_timestamps(df):
    logger.info(f"Merging duplicate timestamps. Initial shape: {df.shape}")
    # Sort the dataframe by timestamp and then by the number of non-null values (descending)
    df['non_null_count'] = df.notna().sum(axis=1)
    df_sorted = df.sort_values(['TIMESTAMP', 'non_null_count'], ascending=[True, False])
    
    # Group by timestamp and merge, keeping the first non-null value for each column
    df_merged = df_sorted.groupby('TIMESTAMP', as_index=False).first()
    
    # Drop the helper column
    df_merged = df_merged.drop(columns=['non_null_count'])
    
    logger.info(f"Merged {len(df) - len(df_merged)} duplicate timestamp rows. Final shape: {df_merged.shape}")
    logger.info(f"Sample of merged data:\n{df_merged.head().to_string()}")
    return df_merged

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
    df = merge_duplicate_timestamps(df)
    
    logger.info(f"{table_name} data range: {df['TIMESTAMP'].min()} to {df['TIMESTAMP'].max()}")
    logger.info(f"{table_name} data shape: {df.shape}")
    logger.info(f"{table_name} columns: {df.columns.tolist()}")
    logger.info(f"Sample of {table_name} data:\n{df.head().to_string()}")
    
    return df

def clip_forecast_data(df, clip_timestamp):
    logger.info(f"Clipping forecast data at {clip_timestamp}. Initial shape: {df.shape}")
    df_former = df[df['TIMESTAMP'] <= clip_timestamp]
    df_latter = df[df['TIMESTAMP'] > clip_timestamp]
    logger.info(f"Clipped data shapes - Former: {df_former.shape}, Latter: {df_latter.shape}")
    return df_former, df_latter

def create_full_hourly_index(start_time, end_time):
    logger.info(f"Creating full hourly index from {start_time} to {end_time}")
    return pd.date_range(start=start_time, end=end_time, freq='h')

def interpolate_hourly(df, full_index):
    logger.info(f"Interpolating hourly data. Initial shape: {df.shape}")
    df = df.set_index('TIMESTAMP')
    
    # Reindex the dataframe to the full hourly index
    df_hourly = df.reindex(full_index)
    
    # Separate datetime columns and numeric columns
    datetime_columns = df_hourly.select_dtypes(include=['datetime64']).columns
    numeric_columns = df_hourly.select_dtypes(include=['float64', 'int64']).columns
    
    # Interpolate numeric columns
    df_hourly[numeric_columns] = df_hourly[numeric_columns].interpolate(method='time')
    
    # Forward fill datetime columns
    df_hourly[datetime_columns] = df_hourly[datetime_columns].ffill()
    
    # Special handling for Rain1mTot
    if 'Rain1mTot' in df_hourly.columns:
        df_hourly['Rain1mTot'] = df_hourly['Rain1mTot'].fillna(0)
    
    # Reset index to make TIMESTAMP a column again
    df_hourly = df_hourly.reset_index()
    df_hourly = df_hourly.rename(columns={'index': 'TIMESTAMP'})
    
    logger.info(f"Interpolated data shape: {df_hourly.shape}")
    logger.info(f"Sample of interpolated data:\n{df_hourly.head().to_string()}")
    return df_hourly

def append_forecast_to_mesonet(mesonet_df, static_former, rolling_former):
    logger.info("Appending forecast data to mesonet data")
    logger.info(f"Input shapes - Mesonet: {mesonet_df.shape}, Static former: {static_former.shape}, Rolling former: {rolling_former.shape}")
    combined_df = pd.concat([mesonet_df, static_former, rolling_former], axis=0)
    combined_df = combined_df.sort_values('TIMESTAMP').drop_duplicates(subset='TIMESTAMP', keep='first')
    logger.info(f"Combined data shape: {combined_df.shape}")
    logger.info(f"Sample of combined data:\n{combined_df.head().to_string()}")
    return combined_df

def align_rolling_latter(rolling_latter, mesonet_latest):
    logger.info(f"Aligning rolling latter data. Initial shape: {rolling_latter.shape}")
    time_diff = rolling_latter['TIMESTAMP'].min() - mesonet_latest
    rolling_latter['TIMESTAMP'] = rolling_latter['TIMESTAMP'] - time_diff
    logger.info(f"Aligned rolling latter data shape: {rolling_latter.shape}")
    logger.info(f"Sample of aligned data:\n{rolling_latter.head().to_string()}")
    return rolling_latter

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
    df = merge_duplicate_timestamps(df)
    
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

def add_weather_data_to_tables(weather_df):
    logger.info(f"Adding weather data to plot tables. Weather data shape: {weather_df.shape}")
    logger.info(f"Weather data columns: {weather_df.columns.tolist()}")
    logger.info(f"Sample of weather data:\n{weather_df.head().to_string()}")

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # Get list of plot tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'plot_%'")
    plot_tables = [row[0] for row in cursor.fetchall()]
    logger.info(f"Found plot tables: {plot_tables}")

    # Convert weather_df TIMESTAMP to datetime
    weather_df['TIMESTAMP'] = pd.to_datetime(weather_df['TIMESTAMP'])

    for table in plot_tables:
        logger.info(f"Processing table: {table}")
        # Get existing columns
        cursor.execute(f"PRAGMA table_info({table})")
        existing_columns = [row[1] for row in cursor.fetchall()]
        logger.info(f"Existing columns in {table}: {existing_columns}")

        # Add weather columns if they don't exist
        for column in weather_df.columns:
            if column != 'TIMESTAMP':
                weather_column = f"weather_{column}"
                if weather_column not in existing_columns:
                    cursor.execute(f"ALTER TABLE {table} ADD COLUMN {weather_column} REAL")
                    logger.info(f"Added new column to {table}: {weather_column}")

        # Fetch all timestamps from the plot table
        cursor.execute(f"SELECT TIMESTAMP FROM {table}")
        plot_timestamps = [row[0] for row in cursor.fetchall()]
        plot_timestamps = pd.to_datetime(plot_timestamps)

        # Find closest weather data for each plot timestamp
        rows_updated = 0
        for plot_timestamp in plot_timestamps:
            closest_weather = weather_df.iloc[weather_df['TIMESTAMP'].sub(plot_timestamp).abs().idxmin()]
            
            update_query = f"""
            UPDATE {table}
            SET {', '.join([f"weather_{col} = ?" for col in weather_df.columns if col != 'TIMESTAMP'])}
            WHERE TIMESTAMP = ?
            """
            
            values = []
            for col in weather_df.columns:
                if col != 'TIMESTAMP':
                    val = closest_weather[col]
                    if pd.isna(val):
                        values.append(None)
                    elif isinstance(val, pd.Timestamp):
                        values.append(val.isoformat())
                    else:
                        values.append(val)
            
            values.append(plot_timestamp.isoformat())
            
            cursor.execute(update_query, values)
            rows_updated += cursor.rowcount

        conn.commit()
        logger.info(f"Updated {rows_updated} rows in table {table}")

        # Verify updates
        cursor.execute(f"SELECT * FROM {table} LIMIT 5")
        sample_data = cursor.fetchall()
        logger.info(f"Sample of updated data in {table}:\n{sample_data}")

    conn.close()
    logger.info("Added weather data to all plot tables")

def main():
    logger.info("Starting MPC data preparation")
    logger.info("Triggering cloud functions...")
    trigger_cloud_functions()

    logger.info("Retrieving and processing weather data...")
    try:
        mesonet_data = get_data_with_history('current-weather-mesonet')
        static_forecast = get_data_with_history('forecast_four_day_static')
        rolling_forecast = get_data_with_history('forecast_four_day_rolling')

        mesonet_latest_timestamp = mesonet_data['TIMESTAMP'].max()
        logger.info(f"Latest mesonet timestamp: {mesonet_latest_timestamp}")
        
        # Create a full hourly index from the earliest to the latest timestamp
        full_index = create_full_hourly_index(min(mesonet_data['TIMESTAMP'].min(), 
                                                  static_forecast['TIMESTAMP'].min(), 
                                                  rolling_forecast['TIMESTAMP'].min()),
                                              max(static_forecast['TIMESTAMP'].max(), 
                                                  rolling_forecast['TIMESTAMP'].max()))
        
        static_former, _ = clip_forecast_data(static_forecast, mesonet_latest_timestamp)
        rolling_former, rolling_latter = clip_forecast_data(rolling_forecast, mesonet_latest_timestamp)

        logger.info("Interpolating hourly data...")
        static_former_hourly = interpolate_hourly(static_former, full_index)
        rolling_former_hourly = interpolate_hourly(rolling_former, full_index)
        rolling_latter_hourly = interpolate_hourly(rolling_latter, full_index)
        mesonet_hourly = interpolate_hourly(mesonet_data, full_index)

        logger.info("Combining data...")
        logger.info(f"Columns in mesonet_hourly: {mesonet_hourly.columns.tolist()}")
        logger.info(f"Columns in static_former_hourly: {static_former_hourly.columns.tolist()}")
        logger.info(f"Columns in rolling_former_hourly: {rolling_former_hourly.columns.tolist()}")
        
        combined_data = append_forecast_to_mesonet(mesonet_hourly, static_former_hourly, rolling_former_hourly)
        aligned_rolling_latter = align_rolling_latter(rolling_latter_hourly, mesonet_latest_timestamp)
        final_combined_data = pd.concat([combined_data, aligned_rolling_latter], axis=0).sort_values('TIMESTAMP')
        final_combined_data = merge_duplicate_timestamps(final_combined_data)  # Final deduplication

        logger.info("Final combined data summary:")
        logger.info(f"Shape: {final_combined_data.shape}")
        logger.info(f"Columns: {final_combined_data.columns.tolist()}")
        logger.info(f"Date range: {final_combined_data['TIMESTAMP'].min()} to {final_combined_data['TIMESTAMP'].max()}")
        logger.info(f"Sample of final combined data:\n{final_combined_data.head().to_string()}")

        logger.info("Creating SQLite database...")
        create_sqlite_db()

        logger.info("Retrieving treatment 1 plots...")
        plots = get_treatment_1_plots()
        logger.info(f"Found {len(plots)} plots: {plots}")

        for plot in plots:
            logger.info(f"Processing plot {plot}...")
            try:
                plot_data = get_plot_data(plot)
                create_plot_table(plot, plot_data)
            except Exception as e:
                logger.error(f"Error processing plot {plot}: {str(e)}")
                continue

        logger.info("Adding weather data to plot tables...")
        add_weather_data_to_tables(final_combined_data)

        logger.info("Data preparation complete.")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()