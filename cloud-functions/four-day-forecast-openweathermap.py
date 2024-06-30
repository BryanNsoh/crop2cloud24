import os
import requests
from google.cloud import bigquery
from datetime import datetime
import pytz
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OpenWeatherMap API details
API_KEY = os.environ.get('OPENWEATHERMAP_API_KEY')
BASE_URL = "https://api.openweathermap.org/data/2.5/forecast"

# BigQuery details
PROJECT_ID = "crop2cloud24"
DATASET_ID = "weather"
TABLE_ID = "4-day-forecast-openweathermap"

def get_forecast(lat, lon):
    params = {
        'lat': lat,
        'lon': lon,
        'appid': API_KEY,
        'units': 'metric'
    }
    logger.info(f"Requesting forecast data for coordinates: {lat}, {lon}")
    response = requests.get(BASE_URL, params=params)
    response.raise_for_status()
    logger.info("Successfully retrieved forecast data")
    return response.json()

def map_forecast_data(forecast_item):
    logger.info("Mapping forecast data to BigQuery schema")
    cst = pytz.timezone('America/Chicago')
    timestamp = datetime.fromtimestamp(forecast_item['dt'], pytz.UTC).astimezone(cst)
    mapped_data = {
        'TIMESTAMP': timestamp.isoformat(),
        'Ta_2m_Avg': forecast_item['main']['temp'],
        'TaMax_2m': forecast_item['main']['temp_max'],
        'TaMin_2m': forecast_item['main']['temp_min'],
        'RH_2m_Avg': forecast_item['main']['humidity'],
        'Dp_2m_Avg': forecast_item['main'].get('dew_point'),
        'WndAveSpd_3m': forecast_item['wind']['speed'],
        'WndAveDir_3m': forecast_item['wind']['deg'],
        'WndMaxSpd5s_3m': forecast_item['wind'].get('gust'),
        'PresAvg_1pnt5m': forecast_item['main']['pressure'],
        'Rain_1m_Tot': forecast_item['rain']['3h'] if 'rain' in forecast_item else 0,
        'UV_index': 0,  # Forecast doesn't include UV index
        'Visibility': forecast_item.get('visibility', 0),
        'Clouds': forecast_item['clouds']['all']
    }
    logger.info(f"Mapped data: {json.dumps(mapped_data, indent=2)}")
    return mapped_data

def ensure_table_exists(client):
    dataset_ref = client.dataset(DATASET_ID)
    table_ref = dataset_ref.table(TABLE_ID)
    
    try:
        client.get_table(table_ref)
        logger.info(f"Table {PROJECT_ID}.{DATASET_ID}.{TABLE_ID} already exists")
    except Exception as e:
        logger.info(f"Table {PROJECT_ID}.{DATASET_ID}.{TABLE_ID} does not exist. Creating it now.")
        schema = [
            bigquery.SchemaField("TIMESTAMP", "TIMESTAMP"),
            bigquery.SchemaField("Ta_2m_Avg", "FLOAT"),
            bigquery.SchemaField("TaMax_2m", "FLOAT"),
            bigquery.SchemaField("TaMin_2m", "FLOAT"),
            bigquery.SchemaField("RH_2m_Avg", "FLOAT"),
            bigquery.SchemaField("Dp_2m_Avg", "FLOAT"),
            bigquery.SchemaField("WndAveSpd_3m", "FLOAT"),
            bigquery.SchemaField("WndAveDir_3m", "FLOAT"),
            bigquery.SchemaField("WndMaxSpd5s_3m", "FLOAT"),
            bigquery.SchemaField("PresAvg_1pnt5m", "FLOAT"),
            bigquery.SchemaField("Rain_1m_Tot", "FLOAT"),
            bigquery.SchemaField("UV_index", "FLOAT"),
            bigquery.SchemaField("Visibility", "FLOAT"),
            bigquery.SchemaField("Clouds", "FLOAT")
        ]
        table = bigquery.Table(table_ref, schema=schema)
        table.time_partitioning = bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.DAY,
            field="TIMESTAMP"
        )
        try:
            client.create_table(table)
            logger.info(f"Created table {PROJECT_ID}.{DATASET_ID}.{TABLE_ID}")
        except Exception as create_error:
            logger.error(f"Error creating table: {str(create_error)}")
            raise

def insert_into_bigquery(data_list):
    client = bigquery.Client()
    ensure_table_exists(client)
    table_ref = client.dataset(DATASET_ID).table(TABLE_ID)
    
    job_config = bigquery.LoadJobConfig()
    job_config.write_disposition = bigquery.WriteDisposition.WRITE_TRUNCATE
    job_config.autodetect = True

    job = client.load_table_from_json(data_list, table_ref, job_config=job_config)
    job.result()  # Wait for the job to complete

    logger.info(f"{len(data_list)} rows have been added/updated successfully.")

def four_day_forecast_openweathermap(request):
    try:
        logger.info("Starting 4-day forecast function")
        lat, lon = 41.089075, -100.773775
        
        forecast_data = get_forecast(lat, lon)
        mapped_data_list = [map_forecast_data(item) for item in forecast_data['list']]
        
        insert_into_bigquery(mapped_data_list)
        
        logger.info("Forecast data processed successfully")
        return 'Forecast data processed successfully', 200
    except Exception as e:
        logger.error(f"Error processing forecast data: {str(e)}", exc_info=True)
        return f'Error processing forecast data: {str(e)}', 500