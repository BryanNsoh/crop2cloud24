import pandas as pd
import numpy as np
from google.cloud import bigquery
from datetime import datetime, timedelta
import pytz
import logging
from ..config import PROJECT_ID, HISTORICAL_DAYS

logger = logging.getLogger(__name__)

# List of columns to exclude (based on columns with many null values in mesonet data)
EXCLUDE_COLUMNS = [
    'TaMaxTime_2m', 'TaMinTime_2m', 'RHMaxTime_2m', 'RHMinTime_2m',
    'DpMaxTime_2m', 'DpMinTime_2m', 'HeatIndexMaxTime_2m',
    'WindChillMinTime_2m', 'WndMaxSpd5sTime_3m', 'PresMaxTime_1pnt5m',
    'PresMinTime_1pnt5m', 'TsMaxTime_bare_10cm', 'TsMinTime_bare_10cm', 'is_forecast', 
    'collection_time', 'BattVolts_Min', 'LithBatt_Min', 'MaintMode', 'UV_index'
]

def get_columns_to_exclude(df, threshold=0.95):
    """
    Identify columns with a high percentage of null values.
    
    Args:
    df (pd.DataFrame): The DataFrame to analyze
    threshold (float): The threshold for null value percentage (default: 0.95)
    
    Returns:
    list: Columns with null value percentage above the threshold
    """
    null_percentages = df.isnull().mean()
    return list(null_percentages[null_percentages > threshold].index)

def get_weather_data(client, table_name):
    end_time = datetime.now(pytz.UTC)
    start_time = end_time - timedelta(days=HISTORICAL_DAYS)
    
    # First, get all columns
    schema_query = f"""
    SELECT column_name
    FROM `{PROJECT_ID}.weather.INFORMATION_SCHEMA.COLUMNS`
    WHERE table_name = '{table_name}'
    """
    schema_job = client.query(schema_query)
    all_columns = [row['column_name'] for row in schema_job]
    
    # Exclude the predefined columns
    columns_to_select = [col for col in all_columns if col not in EXCLUDE_COLUMNS]
    
    columns_string = ", ".join(columns_to_select)
    
    # Adjust the query based on whether it's a forecast table or not
    if 'forecast' in table_name:
        query = f"""
        SELECT {columns_string}
        FROM `{PROJECT_ID}.weather.{table_name}`
        WHERE TIMESTAMP >= '{start_time}'
        ORDER BY TIMESTAMP
        """
    else:
        query = f"""
        SELECT {columns_string}
        FROM `{PROJECT_ID}.weather.{table_name}`
        WHERE TIMESTAMP BETWEEN '{start_time}' AND '{end_time}'
        ORDER BY TIMESTAMP
        """
    
    logger.info(f"Executing query for {table_name}:\n{query}")
    df = client.query(query).to_dataframe()
    
    logger.info(f"Raw data retrieved for {table_name}. Shape: {df.shape}")
    
    # Identify additional columns to exclude based on null percentage
    additional_excludes = get_columns_to_exclude(df)
    logger.info(f"Additional columns excluded due to high null percentage: {additional_excludes}")
    
    # Remove additional columns with high null percentage
    df = df.drop(columns=additional_excludes, errors='ignore')
    
    logger.info(f"Columns with null values: {df.columns[df.isnull().any()].tolist()}")
    logger.info(f"Number of null values per column:\n{df.isnull().sum()}")
    
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
    
    null_timestamps = df['TIMESTAMP'].isnull().sum()
    logger.info(f"Number of null timestamps after conversion: {null_timestamps}")
    
    df = df.dropna(subset=['TIMESTAMP'])
    logger.info(f"Shape after dropping null timestamps: {df.shape}")
    
    logger.info(f"{table_name} data range: {df['TIMESTAMP'].min()} to {df['TIMESTAMP'].max()}")
    logger.info(f"{table_name} data shape: {df.shape}")
    logger.info(f"{table_name} columns: {df.columns.tolist()}")
    logger.info(f"Sample of {table_name} data:\n{df.head().to_string()}")
    
    return df

def get_all_weather_data(client, table_names):
    weather_data = {}
    for table_name in table_names:
        weather_data[table_name] = get_weather_data(client, table_name)
    return weather_data