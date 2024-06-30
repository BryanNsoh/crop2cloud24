import os
import requests
import sqlite3
import pandas as pd
from google.cloud import bigquery
from dotenv import load_dotenv
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()

# BigQuery client
client = bigquery.Client()

# SQLite database
DB_NAME = 'mpc_data.db'

# Cloud function trigger URLs
TRIGGER_URLS = [
    'https://us-central1-crop2cloud24.cloudfunctions.net/four-day-forecast-openweathermap',
    'https://us-central1-crop2cloud24.cloudfunctions.net/compute-cwsi',
    'https://us-central1-crop2cloud24.cloudfunctions.net/compute-swsi',
    'https://us-central1-crop2cloud24.cloudfunctions.net/current-openweathermap',
    'https://us-central1-crop2cloud24.cloudfunctions.net/weather-updater'
]

def trigger_cloud_functions():
    for url in TRIGGER_URLS:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                print(f"Successfully triggered: {url}")
            else:
                print(f"Failed to trigger: {url}. Status code: {response.status_code}")
        except requests.RequestException as e:
            print(f"Error triggering {url}: {e}")

def get_treatment_1_plots():
    query = """
    SELECT DISTINCT plot_number
    FROM `crop2cloud24.LINEAR_CORN_trt1.INFORMATION_SCHEMA.TABLES`
    WHERE table_name LIKE 'plot_%'
    """
    return [row.plot_number for row in client.query(query).result()]

def get_plot_data(plot_number):
    query = f"""
    SELECT *
    FROM `crop2cloud24.LINEAR_CORN_trt1.plot_{plot_number}`
    ORDER BY TIMESTAMP DESC
    LIMIT 1000  -- Adjust this limit as needed
    """
    return client.query(query).to_dataframe()

def get_weather_data():
    current_time = datetime.utcnow()
    four_days_later = current_time + timedelta(days=4)
    
    query = f"""
    SELECT *
    FROM `crop2cloud24.weather.current-weather-mesonet`
    WHERE TIMESTAMP BETWEEN '{current_time.isoformat()}' AND '{four_days_later.isoformat()}'
    ORDER BY TIMESTAMP
    """
    return client.query(query).to_dataframe()

def create_sqlite_db():
    conn = sqlite3.connect(DB_NAME)
    conn.close()
    print(f"Created SQLite database: {DB_NAME}")

def create_plot_table(plot_number, df):
    conn = sqlite3.connect(DB_NAME)
    table_name = f"plot_{plot_number}"
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    conn.close()
    print(f"Created table: {table_name}")

def add_weather_data_to_tables(weather_df):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # Get list of plot tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'plot_%'")
    plot_tables = [row[0] for row in cursor.fetchall()]

    for table in plot_tables:
        # Add weather columns if they don't exist
        for column in weather_df.columns:
            if column != 'TIMESTAMP':
                cursor.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS current_{column} REAL")
                cursor.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS forecast_{column} REAL")

        # Update weather data for each row
        for _, weather_row in weather_df.iterrows():
            timestamp = weather_row['TIMESTAMP']
            update_query = f"""
            UPDATE {table}
            SET {', '.join([f"current_{col} = ?" for col in weather_df.columns if col != 'TIMESTAMP'])}
            WHERE TIMESTAMP <= ?
            """
            cursor.execute(update_query, list(weather_row.drop('TIMESTAMP')) + [timestamp])

            forecast_update_query = f"""
            UPDATE {table}
            SET {', '.join([f"forecast_{col} = ?" for col in weather_df.columns if col != 'TIMESTAMP'])}
            WHERE TIMESTAMP > ? AND TIMESTAMP <= ?
            """
            cursor.execute(forecast_update_query, list(weather_row.drop('TIMESTAMP')) + [timestamp, timestamp + timedelta(hours=1)])

    conn.commit()
    conn.close()
    print("Added weather data to all plot tables")

def main():
    print("Triggering cloud functions...")
    trigger_cloud_functions()

    print("Creating SQLite database...")
    create_sqlite_db()

    print("Retrieving treatment 1 plots...")
    plots = get_treatment_1_plots()

    for plot in plots:
        print(f"Processing plot {plot}...")
        plot_data = get_plot_data(plot)
        create_plot_table(plot, plot_data)

    print("Retrieving weather data...")
    weather_data = get_weather_data()

    print("Adding weather data to plot tables...")
    add_weather_data_to_tables(weather_data)

    print("Data preparation complete.")

if __name__ == "__main__":
    main()