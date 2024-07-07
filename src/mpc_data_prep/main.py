import logging
import requests
from google.cloud import bigquery
from .config import TRIGGER_URLS, PROJECT_ID, WEATHER_TABLES
from .weather.data_retrieval import get_all_weather_data
from .weather.data_processing import process_weather_data
from .weather.database_operations import store_weather_data
from .sensor.data_retrieval import get_all_plot_data
from .sensor.data_processing import process_plot_data
from .sensor.database_operations import store_plot_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

def main():
    logger.info("Starting MPC data preparation")
    
    logger.info("Triggering cloud functions...")
    trigger_cloud_functions()

    client = bigquery.Client(project=PROJECT_ID)

    logger.info("Retrieving and processing weather data...")
    try:
        weather_data = get_all_weather_data(client, WEATHER_TABLES)
        mesonet_data, static_forecast, rolling_forecast = process_weather_data(weather_data)
        store_weather_data(mesonet_data, static_forecast, rolling_forecast)
        logger.info("Weather data processing completed.")
    except Exception as e:
        logger.error(f"Error processing weather data: {str(e)}")
        raise

    logger.info("Retrieving and processing plot data...")
    try:
        plot_data = get_all_plot_data(client)
        processed_plot_data = process_plot_data(plot_data)
        store_plot_data(processed_plot_data)
        logger.info("Plot data processing completed.")
    except Exception as e:
        logger.error(f"Error processing plot data: {str(e)}")
        raise

    logger.info("Data preparation completed.")

if __name__ == "__main__":
    main()