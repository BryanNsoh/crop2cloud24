import requests
import os
import pandas as pd
import numpy as np
from google.cloud import bigquery
from pytz import timezone
from datetime import datetime
import logging

from flask import jsonify

# URL and target file
base_url = "https://data.mesonet.unl.edu/data/north_platte_3sw_beta/latest/sincelast/"
file_to_download = "North_Platte_3SW_Beta_1min.csv"

# BigQuery table details
project_id = "crop2cloud24"
dataset_id = "weather"
table_id = "current-weather-mesonet"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DateTimeConverter:
    @staticmethod
    def to_utc(timestamp):
        central = timezone('America/Chicago')
        if timestamp.tzinfo is None:
            # Assume Central Time if no timezone info
            timestamp = central.localize(timestamp)
        return timestamp.astimezone(timezone('UTC'))

class DataParser:
    def __init__(self, table_name):
        self.table_name = table_name
        self.cwd = "/tmp"  # Use /tmp directory in Cloud Functions

    def parse_weather_csv(self, filename):
        def date_parser(date_string):
            return pd.to_datetime(date_string, format="%Y-%m-%d %H:%M:%S", errors="coerce")

        filename = os.path.join(self.cwd, filename)

        try:
            df = pd.read_csv(
                filename,
                header=1,
                skiprows=[2, 3],
                parse_dates=["TIMESTAMP"],
                date_parser=date_parser,
            )
            logger.info(f"Successfully read CSV file: {filename}")
            print("DataFrame after reading CSV:")
            print(df.head())
        except Exception as e:
            logger.error(f"Error reading CSV file: {filename}. Error: {str(e)}")
            raise

        df = df.rename(columns=lambda x: x.strip())
        df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"], errors="coerce")
        df = df.dropna(subset=["TIMESTAMP"])
        df["TIMESTAMP"] = df["TIMESTAMP"].apply(DateTimeConverter.to_utc)
        df = df.set_index("TIMESTAMP")
        df = df.apply(pd.to_numeric, errors="coerce")

        # Remove columns that are not in the BigQuery table schema
        columns_to_remove = ['BattVolts_Min', 'LithBatt_Min', 'MaintMode']
        df = df.drop(columns=columns_to_remove, errors='ignore')

        print("DataFrame after processing:")
        print(df.head())
        print("Columns:", df.columns.tolist())
        print("Index:", df.index.tolist())

        return df

def ensure_dataset_and_table_exist(client, project_id, dataset_id, table_id, schema):
    dataset_ref = client.dataset(dataset_id, project=project_id)
    try:
        client.get_dataset(dataset_ref)
    except Exception:
        dataset = bigquery.Dataset(dataset_ref)
        dataset = client.create_dataset(dataset)
        logger.info(f"Dataset {dataset_id} created.")

    table_ref = dataset_ref.table(table_id)
    try:
        client.get_table(table_ref)
    except Exception:
        table = bigquery.Table(table_ref, schema=schema)
        table = client.create_table(table)
        logger.info(f"Table {table_id} created.")

def download_and_process_data():
    logger.info("Function execution started")

    try:
        os.makedirs("/tmp", exist_ok=True)
        logger.info("Temporary directory created")
    except Exception as e:
        logger.error(f"Error creating temporary directory. Error: {str(e)}")
        raise

    full_url = f"{base_url}{file_to_download}"
    try:
        r = requests.get(full_url, allow_redirects=True)
        r.raise_for_status()
        logger.info(f"Successfully downloaded file: {file_to_download}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading file: {file_to_download}. Error: {str(e)}")
        raise

    if r.status_code == 200:
        try:
            with open(f"/tmp/{file_to_download}", "wb") as f_out:
                f_out.write(r.content)
            logger.info(f"File saved: {file_to_download}")
        except Exception as e:
            logger.error(f"Error saving file: {file_to_download}. Error: {str(e)}")
            raise

    parser = DataParser(f"{project_id}.{dataset_id}.{table_id}")
    try:
        df = parser.parse_weather_csv(f"/tmp/{file_to_download}")
        logger.info("CSV file parsed successfully")
        logger.info(f"Parsed DataFrame:\n{df.head()}")
    except Exception as e:
        logger.error(f"Error parsing CSV file. Error: {str(e)}")
        raise

    client = bigquery.Client()

    schema = [
        bigquery.SchemaField("TIMESTAMP", "TIMESTAMP"),  # Changed to TIMESTAMP
        bigquery.SchemaField("RECORD", "FLOAT"),
        bigquery.SchemaField("Ta_2m_Avg", "FLOAT"),
        bigquery.SchemaField("TaMax_2m", "FLOAT"),
        bigquery.SchemaField("TaMaxTime_2m", "FLOAT"),
        bigquery.SchemaField("TaMin_2m", "FLOAT"),
        bigquery.SchemaField("TaMinTime_2m", "FLOAT"),
        bigquery.SchemaField("RH_2m_Avg", "FLOAT"),
        bigquery.SchemaField("RHMax_2m", "FLOAT"),
        bigquery.SchemaField("RHMaxTime_2m", "FLOAT"),
        bigquery.SchemaField("RHMin_2m", "FLOAT"),
        bigquery.SchemaField("RHMinTime_2m", "FLOAT"),
        bigquery.SchemaField("Dp_2m_Avg", "FLOAT"),
        bigquery.SchemaField("DpMax_2m", "FLOAT"),
        bigquery.SchemaField("DpMaxTime_2m", "FLOAT"),
        bigquery.SchemaField("DpMin_2m", "FLOAT"),
        bigquery.SchemaField("DpMinTime_2m", "FLOAT"),
        bigquery.SchemaField("HeatIndex_2m_Avg", "FLOAT"),
        bigquery.SchemaField("HeatIndexMax_2m", "FLOAT"),
        bigquery.SchemaField("HeatIndexMaxTime_2m", "FLOAT"),
        bigquery.SchemaField("WindChill_2m_Avg", "FLOAT"),
        bigquery.SchemaField("WindChillMin_2m", "FLOAT"),
        bigquery.SchemaField("WindChillMinTime_2m", "FLOAT"),
        bigquery.SchemaField("WndAveSpd_3m", "FLOAT"),
        bigquery.SchemaField("WndVecMagAve_3m", "FLOAT"),
        bigquery.SchemaField("WndAveDir_3m", "FLOAT"),
        bigquery.SchemaField("WndAveDirSD_3m", "FLOAT"),
        bigquery.SchemaField("WndMaxSpd5s_3m", "FLOAT"),
        bigquery.SchemaField("WndMaxSpd5sTime_3m", "FLOAT"),
        bigquery.SchemaField("WndMax_5sec_Dir_3m", "FLOAT"),
        bigquery.SchemaField("PresAvg_1pnt5m", "FLOAT"),
        bigquery.SchemaField("PresMax_1pnt5m", "FLOAT"),
        bigquery.SchemaField("PresMaxTime_1pnt5m", "FLOAT"),
        bigquery.SchemaField("PresMin_1pnt5m", "FLOAT"),
        bigquery.SchemaField("PresMinTime_1pnt5m", "FLOAT"),
        bigquery.SchemaField("Solar_2m_Avg", "FLOAT"),
        bigquery.SchemaField("Rain_1m_Tot", "FLOAT"),
        bigquery.SchemaField("Ts_bare_10cm_Avg", "FLOAT"),
        bigquery.SchemaField("TsMax_bare_10cm", "FLOAT"),
        bigquery.SchemaField("TsMaxTime_bare_10cm", "FLOAT"),
        bigquery.SchemaField("TsMin_bare_10cm", "FLOAT"),
        bigquery.SchemaField("TsMinTime_bare_10cm", "FLOAT"),
    ]

    # Ensure dataset and table exist
    ensure_dataset_and_table_exist(client, project_id, dataset_id, table_id, schema)

    job_config = bigquery.LoadJobConfig(
        schema=schema,
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
    )

    try:
        job = client.load_table_from_dataframe(
            df.reset_index(),  # Reset index to include TIMESTAMP as a column
            f"{project_id}.{dataset_id}.{table_id}",
            job_config=job_config,
        )
        job.result()
        logger.info("Data loaded into BigQuery table successfully")
    except Exception as e:
        logger.error(f"Error loading data into BigQuery table. Error: {str(e)}")
        raise

    logger.info("Function execution completed")

def entry_point(request):
    try:
        download_and_process_data()
        return jsonify({"message": "Data processed successfully"}), 200
    except Exception as e:
        logger.error(f"Error in entry_point function: {str(e)}")
        return jsonify({"error": str(e)}), 500