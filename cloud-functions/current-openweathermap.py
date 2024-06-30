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
BASE_URL = "https://api.openweathermap.org/data/2.5/weather"

# BigQuery details
PROJECT_ID = "crop2cloud24"
DATASET_ID = "weather"
TABLE_ID = "current-openweathermap"

def get_current_weather(lat, lon):
    params = {
        'lat': lat,
        'lon': lon,
        'appid': API_KEY,
        'units': 'metric'
    }
    logger.info(f"Requesting current weather data for coordinates: {lat}, {lon}")
    response = requests.get(BASE_URL, params=params)
    response.raise_for_status()
    logger.info("Successfully retrieved current weather data")
    return response.json()

def map_weather_data(data):
    logger.info("Mapping weather data to BigQuery schema")
    cst = pytz.timezone('America/Chicago')
    timestamp = datetime.now(pytz.UTC).astimezone(cst)
    mapped_data = {
        'TIMESTAMP': timestamp.isoformat(),
        'Ta_2m_Avg': data['main']['temp'],
        'TaMax_2m': data['main']['temp_max'],
        'TaMin_2m': data['main']['temp_min'],
        'RH_2m_Avg': data['main']['humidity'],
        'Dp_2m_Avg': data['main'].get('dew_point'),
        'WndAveSpd_3m': data['wind']['speed'],
        'WndAveDir_3m': data['wind']['deg'],
        'WndMaxSpd5s_3m': data['wind'].get('gust'),
        'PresAvg_1pnt5m': data['main']['pressure'],
        'Rain_1m_Tot': data['rain']['1h'] if 'rain' in data else 0,
        'UV_index': data.get('uvi', 0),
        'Visibility': data['visibility'],
        'Clouds': data['clouds']['all']
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
        try:
            client.create_table(table)
            logger.info(f"Created table {PROJECT_ID}.{DATASET_ID}.{TABLE_ID}")
        except Exception as create_error:
            logger.error(f"Error creating table: {str(create_error)}")
            raise

def insert_into_bigquery(data):
    client = bigquery.Client()
    ensure_table_exists(client)
    table_ref = client.dataset(DATASET_ID).table(TABLE_ID)
    
    errors = client.insert_rows_json(table_ref, [data])
    if errors:
        logger.error(f"Errors inserting rows: {errors}")
    else:
        logger.info("New row has been added successfully.")

def current_weather_function(request):
    try:
        logger.info("Starting current weather function")
        lat, lon = 41.089075, -100.773775
        
        weather_data = get_current_weather(lat, lon)
        mapped_data = map_weather_data(weather_data)
        insert_into_bigquery(mapped_data)
        
        logger.info("Current weather data processed successfully")
        return 'Current weather data processed successfully', 200
    except Exception as e:
        logger.error(f"Error processing current weather data: {str(e)}", exc_info=True)
        return f'Error processing current weather data: {str(e)}', 500