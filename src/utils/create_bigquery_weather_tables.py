import os
import time
from google.cloud import bigquery
from google.api_core import exceptions
from dotenv import load_dotenv
import pandas as pd
import pytz
from datetime import datetime, timedelta
from logger import get_logger

# Load environment variables
load_dotenv()

# Set up logging
logger = get_logger(__name__)

# BigQuery details
PROJECT_ID = "crop2cloud24"
DATASET_ID = "weather"
TABLE_ID = "current-weather-mesonet"

# Local file path (update this to your local file path)
LOCAL_CSV_PATH = r"C:\Users\bnsoh2\Downloads\North_Platte_3SW_Beta_1min (7).csv"

# Number of days of data to keep
DAYS_TO_KEEP = 30

# Columns to exclude
EXCLUDE_COLUMNS = [
    'TaMaxTime_2m', 'TaMinTime_2m', 'RHMaxTime_2m', 'RHMinTime_2m',
    'DpMaxTime_2m', 'DpMinTime_2m', 'HeatIndexMaxTime_2m',
    'WindChillMinTime_2m', 'WndMaxSpd5sTime_3m', 'PresMaxTime_1pnt5m',
    'PresMinTime_1pnt5m', 'TsMaxTime_bare_10cm', 'TsMinTime_bare_10cm', 'is_forecast', 
    'collection_time', 'BattVolts_Min', 'LithBatt_Min', 'MaintMode'
]

def create_bigquery_client():
    credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    if not credentials_path:
        raise ValueError("GOOGLE_APPLICATION_CREDENTIALS not set in .env file")
    return bigquery.Client()

def create_or_replace_table(client):
    dataset_ref = client.dataset(DATASET_ID)
    table_ref = dataset_ref.table(TABLE_ID)

    schema = [
        bigquery.SchemaField("TIMESTAMP", "TIMESTAMP"),
        bigquery.SchemaField("RECORD", "FLOAT"),
        bigquery.SchemaField("Ta_2m_Avg", "FLOAT"),
        bigquery.SchemaField("TaMax_2m", "FLOAT"),
        bigquery.SchemaField("TaMin_2m", "FLOAT"),
        bigquery.SchemaField("RH_2m_Avg", "FLOAT"),
        bigquery.SchemaField("RHMax_2m", "FLOAT"),
        bigquery.SchemaField("RHMin_2m", "FLOAT"),
        bigquery.SchemaField("Dp_2m_Avg", "FLOAT"),
        bigquery.SchemaField("DpMax_2m", "FLOAT"),
        bigquery.SchemaField("DpMin_2m", "FLOAT"),
        bigquery.SchemaField("HeatIndex_2m_Avg", "FLOAT"),
        bigquery.SchemaField("HeatIndexMax_2m", "FLOAT"),
        bigquery.SchemaField("WindChill_2m_Avg", "FLOAT"),
        bigquery.SchemaField("WindChillMin_2m", "FLOAT"),
        bigquery.SchemaField("WndAveSpd_3m", "FLOAT"),
        bigquery.SchemaField("WndVecMagAve_3m", "FLOAT"),
        bigquery.SchemaField("WndAveDir_3m", "FLOAT"),
        bigquery.SchemaField("WndAveDirSD_3m", "FLOAT"),
        bigquery.SchemaField("WndMaxSpd5s_3m", "FLOAT"),
        bigquery.SchemaField("WndMax_5sec_Dir_3m", "FLOAT"),
        bigquery.SchemaField("PresAvg_1pnt5m", "FLOAT"),
        bigquery.SchemaField("PresMax_1pnt5m", "FLOAT"),
        bigquery.SchemaField("PresMin_1pnt5m", "FLOAT"),
        bigquery.SchemaField("Solar_2m_Avg", "FLOAT"),
        bigquery.SchemaField("Rain_1m_Tot", "FLOAT"),
        bigquery.SchemaField("Ts_bare_10cm_Avg", "FLOAT"),
        bigquery.SchemaField("TsMax_bare_10cm", "FLOAT"),
        bigquery.SchemaField("TsMin_bare_10cm", "FLOAT"),
    ]

    table = bigquery.Table(table_ref, schema=schema)
    table.time_partitioning = bigquery.TimePartitioning(
        type_=bigquery.TimePartitioningType.DAY,
        field="TIMESTAMP"
    )

    # Check if the table exists
    try:
        client.get_table(table_ref)
        # If we reach here, the table exists. Delete it.
        client.delete_table(table_ref)
        logger.info(f"Existing table {PROJECT_ID}.{DATASET_ID}.{TABLE_ID} deleted.")
    except exceptions.NotFound:
        # Table doesn't exist, which is fine
        pass

    # Create the new table
    table = client.create_table(table)
    logger.info(f"Table {table.project}.{table.dataset_id}.{table.table_id} created.")

def process_and_upload_csv(client):
    cutoff_date = datetime.now(pytz.UTC) - timedelta(days=DAYS_TO_KEEP)
    cutoff_date = cutoff_date.replace(tzinfo=None)  # Make cutoff_date timezone-naive

    try:
        df = pd.read_csv(
            LOCAL_CSV_PATH,
            header=1,
            skiprows=[2, 3],
            parse_dates=["TIMESTAMP"],
            date_format="%Y-%m-%d %H:%M:%S",
            chunksize=10000  # Process the file in chunks to avoid memory issues
        )
        logger.info(f"Successfully opened CSV file: {LOCAL_CSV_PATH}")
    except Exception as e:
        logger.error(f"Error reading CSV file: {LOCAL_CSV_PATH}. Error: {str(e)}")
        raise

    total_rows_inserted = 0

    for chunk in df:
        chunk = chunk.rename(columns=lambda x: x.strip())
        chunk["TIMESTAMP"] = pd.to_datetime(chunk["TIMESTAMP"], errors="coerce")
        chunk = chunk.dropna(subset=["TIMESTAMP"])

        # Remove excluded columns
        chunk = chunk.drop(columns=EXCLUDE_COLUMNS, errors='ignore')

        # Remove columns that are not in the BigQuery table schema
        columns_to_remove = ['BattVolts_Min', 'LithBatt_Min', 'MaintMode']
        chunk = chunk.drop(columns=columns_to_remove, errors='ignore')

        # Convert all columns (except TIMESTAMP) to numeric, coercing errors to NaN
        for col in chunk.columns:
            if col != "TIMESTAMP":
                chunk[col] = pd.to_numeric(chunk[col], errors='coerce')

        # Filter for last 30 days
        chunk = chunk[chunk["TIMESTAMP"] >= cutoff_date]

        if not chunk.empty:
            # Convert timestamps to strings in ISO format for BigQuery
            chunk["TIMESTAMP"] = chunk["TIMESTAMP"].dt.strftime("%Y-%m-%d %H:%M:%S")
            
            # Replace NaN values with None for BigQuery compatibility
            chunk = chunk.where(pd.notnull(chunk), None)
            
            # Convert to records, explicitly replacing any remaining NaNs
            rows_to_insert = [{k: (v if pd.notnull(v) else None) for k, v in row.items()} 
                              for row in chunk.to_dict('records')]

            table_ref = client.dataset(DATASET_ID).table(TABLE_ID)
            errors = client.insert_rows_json(table_ref, rows_to_insert)
            if errors:
                logger.error(f"Encountered errors while inserting rows: {errors}")
            else:
                total_rows_inserted += len(rows_to_insert)
                logger.info(f"Inserted {len(rows_to_insert)} rows. Total rows inserted: {total_rows_inserted}")

    logger.info(f"Finished processing. Total rows inserted: {total_rows_inserted}")

def main():
    client = create_bigquery_client()
    
    #create_or_replace_table(client)
    
    # Add a 60-second delay between table creation and data insertion
    logger.info("Waiting for 60 seconds to ensure table creation is fully processed...")
    #time.sleep(60)
    
    process_and_upload_csv(client)

if __name__ == "__main__":
    main()