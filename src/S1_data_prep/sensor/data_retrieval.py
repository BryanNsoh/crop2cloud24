from google.cloud import bigquery
import pandas as pd
import logging
from datetime import datetime, timedelta
import pytz

from ..config import PROJECT_ID, HISTORICAL_DAYS

logger = logging.getLogger(__name__)

def get_treatment_datasets(client):
    query = f"""
    SELECT schema_name
    FROM `{PROJECT_ID}.INFORMATION_SCHEMA.SCHEMATA`
    WHERE schema_name LIKE 'LINEAR_CORN_trt%' OR schema_name LIKE 'LINEAR_SOYBEAN_trt%'
    """
    logger.info(f"Executing query to get treatment datasets:\n{query}")
    datasets = [row.schema_name for row in client.query(query).result()]
    logger.info(f"Found treatment datasets: {datasets}")
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

def get_all_plot_data(client):
    all_data = {}
    
    datasets = get_treatment_datasets(client)
    for dataset in datasets:
        crop_type = 'LINEAR_CORN' if 'CORN' in dataset else 'LINEAR_SOYBEAN'
        treatment = dataset.split('_')[-1]  # e.g., 'trt1'
        plot_numbers = get_treatment_plots(client, dataset)
        if crop_type not in all_data:
            all_data[crop_type] = {}
        if treatment not in all_data[crop_type]:
            all_data[crop_type][treatment] = {}
        for plot_number in plot_numbers:
            all_data[crop_type][treatment][plot_number] = get_plot_data(client, dataset, plot_number)
    
    return all_data