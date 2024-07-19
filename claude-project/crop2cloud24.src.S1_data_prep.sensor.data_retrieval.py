# File: crop2cloud24/src/S1_data_prep/sensor/data_retrieval.py

import pandas as pd
import numpy as np
from google.cloud import bigquery
import logging
from datetime import datetime, timedelta
import pytz

from ..config import PROJECT_ID, HISTORICAL_DAYS

logger = logging.getLogger(__name__)

def get_corn_treatment_datasets(client):
    query = f"""
    SELECT schema_name
    FROM `{PROJECT_ID}.INFORMATION_SCHEMA.SCHEMATA`
    WHERE schema_name LIKE 'LINEAR_CORN_trt%'
    """
    logger.info(f"Executing query to get corn treatment datasets:\n{query}")
    datasets = [row.schema_name for row in client.query(query).result()]
    logger.info(f"Found corn treatment datasets: {datasets}")
    return sorted(datasets)

def get_treatment_plots(client, dataset):
    query = f"""
    SELECT table_name
    FROM `{PROJECT_ID}.{dataset}.INFORMATION_SCHEMA.TABLES`
    WHERE table_name LIKE 'plot_%'
    """
    logger.info(f"Executing query to get treatment plots for {dataset}:\n{query}")
    tables = [row.table_name for row in client.query(query).result()]
    plot_numbers = [int(table_name.split('_')[1]) for table_name in tables]
    logger.info(f"Found plot numbers for {dataset}: {plot_numbers}")
    return sorted(plot_numbers)

def get_plot_data(client, dataset, plot_number):
    end_time = datetime.now(pytz.UTC)
    start_time = end_time - timedelta(days=HISTORICAL_DAYS)
    
    query = f"""
    SELECT *
    FROM `{PROJECT_ID}.{dataset}.plot_{plot_number}`
    WHERE TIMESTAMP BETWEEN '{start_time}' AND '{end_time}'
    ORDER BY TIMESTAMP
    """
    
    logger.info(f"Executing query for {dataset}.plot_{plot_number}:\n{query}")
    df = client.query(query).to_dataframe()
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], utc=True)
    
    logger.info(f"{dataset}.plot_{plot_number} data range: {df['TIMESTAMP'].min()} to {df['TIMESTAMP'].max()}")
    logger.info(f"{dataset}.plot_{plot_number} data shape: {df.shape}")
    logger.info(f"{dataset}.plot_{plot_number} columns: {df.columns.tolist()}")
    logger.info(f"Sample of {dataset}.plot_{plot_number} data:\n{df.head().to_string()}")
    
    return df

def get_weather_data(client):
    end_time = datetime.now(pytz.UTC)
    start_time = end_time - timedelta(days=HISTORICAL_DAYS)
    
    weather_tables = ['current-weather-mesonet', 'current-openweathermap', 'four-day-forecast-openweathermap']
    weather_data = {}
    
    for table in weather_tables:
        query = f"""
        SELECT *
        FROM `{PROJECT_ID}.weather.{table}`
        WHERE TIMESTAMP BETWEEN '{start_time}' AND '{end_time}'
        ORDER BY TIMESTAMP
        """
        
        logger.info(f"Executing query for weather.{table}:\n{query}")
        df = client.query(query).to_dataframe()
        df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], utc=True)
        
        logger.info(f"weather.{table} data range: {df['TIMESTAMP'].min()} to {df['TIMESTAMP'].max()}")
        logger.info(f"weather.{table} data shape: {df.shape}")
        logger.info(f"weather.{table} columns: {df.columns.tolist()}")
        logger.info(f"Sample of weather.{table} data:\n{df.head().to_string()}")
        
        weather_data[table] = df
    
    return weather_data

def get_all_data(client):
    all_data = {'plot_data': {}, 'weather_data': {}}
    
    # Get plot data for all corn treatments
    corn_datasets = get_corn_treatment_datasets(client)
    for dataset in corn_datasets:
        plot_numbers = get_treatment_plots(client, dataset)
        all_data['plot_data'][dataset] = {}
        for plot_number in plot_numbers:
            all_data['plot_data'][dataset][plot_number] = get_plot_data(client, dataset, plot_number)
    
    # Get weather data
    all_data['weather_data'] = get_weather_data(client)
    
    return all_data