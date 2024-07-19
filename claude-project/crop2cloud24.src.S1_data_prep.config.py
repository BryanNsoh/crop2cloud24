import os
from dotenv import load_dotenv

load_dotenv()

# BigQuery client
PROJECT_ID = "crop2cloud24"

# SQLite database
DB_NAME = 'mpc_data.db'

# Specify the number of historical days to retrieve
HISTORICAL_DAYS = 30

# Cloud function trigger URLs
TRIGGER_URLS = [
    # 'https://us-central1-crop2cloud24.cloudfunctions.net/compute-cwsi',
    # 'https://us-central1-crop2cloud24.cloudfunctions.net/compute-swsi',
    # 'https://us-central1-crop2cloud24.cloudfunctions.net/current-openweathermap',
    # 'https://us-central1-crop2cloud24.cloudfunctions.net/weather-updater',
    # 'https://us-central1-crop2cloud24.cloudfunctions.net/forecast_four_day_rolling',
]

# Weather table names
WEATHER_TABLES = [
    'current-weather-mesonet',
    'forecast_four_day_rolling'
]
