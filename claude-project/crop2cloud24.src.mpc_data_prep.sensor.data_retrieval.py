import pandas as pd
import numpy as np
from google.cloud import bigquery
import logging
from datetime import datetime, timedelta
import pytz

from ..config import PROJECT_ID, TREATMENT_1_DATASET, HISTORICAL_DAYS

logger = logging.getLogger(__name__)

def get_treatment_1_plots(client):
    query = f"""
    SELECT table_name
    FROM `{PROJECT_ID}.{TREATMENT_1_DATASET}.INFORMATION_SCHEMA.TABLES`
    WHERE table_name LIKE 'plot_%'
    """
    logger.info(f"Executing query to get treatment 1 plots:\n{query}")
    tables = [row.table_name for row in client.query(query).result()]
    plot_numbers = [int(table_name.split('_')[1]) for table_name in tables]
    logger.info(f"Found plot numbers: {plot_numbers}")
    return sorted(plot_numbers)

def get_plot_data(client, plot_number):
    end_time = datetime.now(pytz.UTC)
    start_time = end_time - timedelta(days=HISTORICAL_DAYS)
    
    query = f"""
    SELECT *
    FROM `{PROJECT_ID}.{TREATMENT_1_DATASET}.plot_{plot_number}`
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

def get_all_plot_data(client):
    plot_numbers = get_treatment_1_plots(client)
    plot_data = {}
    for plot_number in plot_numbers:
        plot_data[plot_number] = get_plot_data(client, plot_number)
    return plot_data