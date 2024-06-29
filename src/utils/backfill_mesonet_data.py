import os
import pandas as pd
import numpy as np
from google.cloud import bigquery
from pytz import timezone
from datetime import datetime, timedelta
import logging
from dateutil.parser import parse

# BigQuery table details
project_id = "crop2cloud24"
dataset_id = "weather"
table_id = "current-weather-mesonet"

# Specify the path to your local CSV file here
CSV_PATH = r"C:\Users\bnsoh2\Downloads\North_Platte_3SW_Beta_1min (3).csv"

# Specify the start date for backfilling here (format: YYYY-MM-DD)
START_DATE = "2024-06-01"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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
    def parse_weather_csv(self, filename):
        def date_parser(date_string):
            return pd.to_datetime(date_string, format="%Y-%m-%d %H:%M:%S", errors="coerce")

        try:
            df = pd.read_csv(
                filename,
                header=1,
                skiprows=[2, 3],
                parse_dates=["TIMESTAMP"],
                date_parser=date_parser,
            )
            logger.info(f"Successfully read CSV file: {filename}")
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

        return df

def get_earliest_timestamp(client, project_id, dataset_id, table_id):
    query = f"""
    SELECT MIN(TIMESTAMP) as earliest_timestamp
    FROM `{project_id}.{dataset_id}.{table_id}`
    """
    query_job = client.query(query)
    results = query_job.result()
    for row in results:
        return row.earliest_timestamp
    return None

def backfill_data():
    client = bigquery.Client()

    # Get the earliest timestamp from the BigQuery table
    earliest_timestamp = get_earliest_timestamp(client, project_id, dataset_id, table_id)
    if earliest_timestamp is None:
        logger.error("Unable to retrieve earliest timestamp from BigQuery table")
        return

    logger.info(f"Earliest timestamp in BigQuery table: {earliest_timestamp}")

    # Parse the start date
    start_date = parse(START_DATE).replace(tzinfo=timezone('UTC'))
    
    # Ensure start_date is before earliest_timestamp
    if start_date >= earliest_timestamp:
        logger.error("Start date must be before the earliest timestamp in the BigQuery table")
        return

    # Parse the CSV file
    parser = DataParser()
    df = parser.parse_weather_csv(CSV_PATH)

    # Filter the dataframe to include only the data we want to backfill
    df_to_insert = df[(df.index >= start_date) & (df.index < earliest_timestamp)]

    if df_to_insert.empty:
        logger.info("No data to backfill within the specified date range")
        return

    logger.info(f"Preparing to insert {len(df_to_insert)} rows")

    # Prepare the data for insertion
    df_to_insert = df_to_insert.reset_index()

    # Define the schema
    schema = [
        bigquery.SchemaField("TIMESTAMP", "TIMESTAMP"),
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

    # Configure the load job
    job_config = bigquery.LoadJobConfig(
        schema=schema,
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
    )

    # Load the data into BigQuery
    table_ref = client.dataset(dataset_id).table(table_id)
    job = client.load_table_from_dataframe(df_to_insert, table_ref, job_config=job_config)

    try:
        job.result()  # Wait for the job to complete
        logger.info(f"Successfully inserted {job.output_rows} rows into BigQuery table")
    except Exception as e:
        logger.error(f"Error inserting data into BigQuery: {str(e)}")

if __name__ == "__main__":
    backfill_data()