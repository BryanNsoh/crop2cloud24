import os
import yaml
from google.cloud import bigquery
from google.oauth2 import service_account
from google.api_core import exceptions
from dotenv import load_dotenv
from logger import get_logger
import pandas as pd
from datetime import datetime
import pytz

logger = get_logger(__name__)

def load_sensor_mapping(file_path='sensor_mapping.yaml'):
    try:
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        logger.error(f"Sensor mapping file not found: {file_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {e}")
        raise

def create_bigquery_client():
    load_dotenv()
    credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    
    if not credentials_path:
        logger.error("GOOGLE_APPLICATION_CREDENTIALS not set in .env file")
        raise ValueError("GOOGLE_APPLICATION_CREDENTIALS not set in .env file")
    
    try:
        credentials = service_account.Credentials.from_service_account_file(
            credentials_path,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        return bigquery.Client(credentials=credentials, project=credentials.project_id)
    except Exception as e:
        logger.error(f"Error creating BigQuery client: {e}")
        raise

def ensure_dataset_exists(client, dataset_id):
    dataset_ref = client.dataset(dataset_id)
    try:
        client.get_dataset(dataset_ref)
        logger.info(f"Dataset {dataset_id} already exists")
    except exceptions.NotFound:
        dataset = bigquery.Dataset(dataset_ref)
        dataset = client.create_dataset(dataset)
        logger.info(f"Created dataset {dataset_id}")

def get_table_schema(client, table_id):
    try:
        table = client.get_table(table_id)
        return {field.name: field.field_type for field in table.schema}
    except exceptions.NotFound:
        return {}

def update_table_schema(client, table_id, new_columns):
    table = client.get_table(table_id)
    original_schema = table.schema
    new_schema = original_schema[:]

    for col_name, col_type in new_columns.items():
        if col_name not in [field.name for field in original_schema]:
            new_schema.append(bigquery.SchemaField(col_name, col_type, mode="NULLABLE"))

    if new_schema != original_schema:
        table.schema = new_schema
        client.update_table(table, ["schema"])
        logger.info(f"Updated schema for table {table_id}")

def get_latest_actual_timestamp(client, table_id):
    query = f"""
    SELECT MAX(TIMESTAMP) as latest_timestamp
    FROM `{table_id}`
    WHERE is_actual = TRUE
    """
    query_job = client.query(query)
    results = query_job.result()
    for row in results:
        if row.latest_timestamp:
            return row.latest_timestamp
    return None

def insert_or_update_data(client, table_id, df, is_actual=True):
    logger.info(f"Preparing to {'insert' if is_actual else 'update'} data in {table_id}")
    logger.info(f"DataFrame columns: {df.columns.tolist()}")
    logger.info(f"DataFrame shape: {df.shape}")

    # Ensure TIMESTAMP is in Central Time
    chicago_tz = pytz.timezone('America/Chicago')
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP']).dt.tz_convert(chicago_tz)

    # Add is_actual column
    df['is_actual'] = is_actual

    if is_actual:
        # Get the latest actual timestamp from the table (already in Central Time)
        latest_timestamp = get_latest_actual_timestamp(client, table_id)
        if latest_timestamp:
            df = df[df['TIMESTAMP'] > latest_timestamp]
            logger.info(f"Filtered data to {len(df)} new rows after {latest_timestamp}")
    else:
        # For predictions, delete existing predictions before inserting new ones
        delete_query = f"""
        DELETE FROM `{table_id}`
        WHERE is_actual = FALSE AND TIMESTAMP >= '{df['TIMESTAMP'].min()}'
        """
        client.query(delete_query).result()
        logger.info(f"Deleted existing predictions from {df['TIMESTAMP'].min()}")

    if df.empty:
        logger.info(f"No new data to {'insert' if is_actual else 'update'} in {table_id}")
        return

    # Sort the dataframe by timestamp
    df = df.sort_values('TIMESTAMP')

    # Get current schema
    current_schema = get_table_schema(client, table_id)
    logger.info(f"Current schema for table {table_id}: {current_schema}")

    # Identify new columns
    new_columns = {col: 'FLOAT64' for col in df.columns if col not in current_schema and col not in ['TIMESTAMP', 'is_actual']}

    # Update schema if new columns exist
    if new_columns:
        logger.info(f"New columns to be added: {new_columns}")
        update_table_schema(client, table_id, new_columns)

    # Ensure we only upload columns that exist in the schema
    columns_to_upload = [col for col in df.columns if col in current_schema or col in new_columns]
    df = df[columns_to_upload]

    # Prepare job config
    job_config = bigquery.LoadJobConfig(
        schema=[
            bigquery.SchemaField('TIMESTAMP', 'TIMESTAMP', mode='REQUIRED'),
            bigquery.SchemaField('is_actual', 'BOOLEAN', mode='REQUIRED')
        ] + [
            bigquery.SchemaField(col, 'FLOAT64', mode='NULLABLE') 
            for col in columns_to_upload if col not in ['TIMESTAMP', 'is_actual']
        ],
        write_disposition="WRITE_APPEND",
    )

    # Upload data
    try:
        job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
        job.result()  # Wait for the job to complete
        logger.info(f"Successfully loaded {len(df)} rows into {table_id}")
    except exceptions.BadRequest as e:
        logger.error(f"Error uploading data to {table_id}: {str(e)}")
        raise

def verify_bigquery_data(client, table_id, df):
    logger.info(f"Verifying data in {table_id}")

    # Query to get the count of rows and the min/max timestamps
    query = f"""
    SELECT 
        COUNT(*) as row_count,
        MIN(TIMESTAMP) as min_timestamp,
        MAX(TIMESTAMP) as max_timestamp,
        SUM(CASE WHEN is_actual THEN 1 ELSE 0 END) as actual_count,
        SUM(CASE WHEN NOT is_actual THEN 1 ELSE 0 END) as prediction_count
    FROM `{table_id}`
    """
    query_job = client.query(query)
    results = query_job.result()

    for row in results:
        logger.info(f"Table {table_id} contains {row.row_count} rows")
        logger.info(f"Timestamp range: {row.min_timestamp} to {row.max_timestamp}")
        logger.info(f"Actual readings: {row.actual_count}")
        logger.info(f"Predictions: {row.prediction_count}")

    # Compare with the original dataframe
    logger.info(f"Original dataframe contains {len(df)} rows")
    logger.info(f"Original timestamp range: {df['TIMESTAMP'].min()} to {df['TIMESTAMP'].max()}")

    # Query to get a sample of data for manual inspection
    sample_query = f"""
    SELECT *
    FROM `{table_id}`
    ORDER BY TIMESTAMP DESC
    LIMIT 10
    """
    sample_job = client.query(sample_query)
    sample_results = sample_job.result()

    logger.info("Sample of uploaded data:")
    for row in sample_results:
        logger.info(row)

def process_and_upload_data(df, sensor_mapping, is_actual=True):
    client = create_bigquery_client()
    
    # Group sensors by treatment and plot
    sensor_groups = {}
    for sensor in sensor_mapping:
        key = (sensor['treatment'], sensor['plot_number'])
        if key not in sensor_groups:
            sensor_groups[key] = []
        sensor_groups[key].append(sensor['sensor_id'])

    # Process and upload data for each group
    for (treatment, plot_number), sensors in sensor_groups.items():
        table_id = f"LINEAR_CORN_trt{treatment}.plot_{plot_number}"
        dataset_id = f"LINEAR_CORN_trt{treatment}"
        ensure_dataset_exists(client, dataset_id)
        full_table_id = f"{client.project}.{dataset_id}.plot_{plot_number}"
        
        # Select relevant columns for this group
        columns_to_upload = ['TIMESTAMP'] + [s for s in sensors if s in df.columns]
        df_to_upload = df[columns_to_upload].dropna(subset=columns_to_upload[1:], how='all')
        
        if not df_to_upload.empty:
            try:
                insert_or_update_data(client, full_table_id, df_to_upload, is_actual)
                verify_bigquery_data(client, full_table_id, df_to_upload)
            except Exception as e:
                logger.error(f"Failed to upload or verify data for plot {plot_number} to {full_table_id}: {str(e)}")
        else:
            logger.info(f"No data to upload for plot {plot_number}")

    return sensor_groups