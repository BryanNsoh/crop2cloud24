import os
import pandas as pd
import numpy as np
from google.cloud import bigquery
from google.api_core.exceptions import NotFound
from logger import get_logger
from bigquery_operations import load_sensor_mapping, create_bigquery_client

logger = get_logger(__name__)

def parse_dat_file(file_name):
    logger.info(f"Parsing file: {file_name}")
    with open(file_name, "r") as file:
        lines = file.readlines()
    headers = lines[1].strip().split(",")
    data_lines = lines[4:]
    data = pd.DataFrame([line.strip().split(",") for line in data_lines], columns=headers)
    
    data.columns = data.columns.str.replace('"', "").str.replace("RECORD", "RecNbr")
    data.columns = data.columns.str.replace("_Avg", "")
    data = data.replace({"NAN": np.nan, '"NAN"': np.nan})
    data["TIMESTAMP"] = data["TIMESTAMP"].str.replace('"', "")
    
    logger.info(f"Original columns: {data.columns.tolist()}")

    # Convert columns to numeric, ignoring errors
    for col in data.columns:
        if col != "TIMESTAMP":
            data[col] = pd.to_numeric(data[col], errors='coerce')
    
    data["TIMESTAMP"] = pd.to_datetime(data["TIMESTAMP"], errors="coerce")
    logger.info(f"Number of rows before dropping NaT timestamps: {len(data)}")
    data = data[~data["TIMESTAMP"].isna()]
    logger.info(f"Number of rows after dropping NaT timestamps: {len(data)}")
    
    data = data.set_index("TIMESTAMP")

    data.index = data.index.tz_localize("America/Chicago")
    
    # Correct the misspelled column names
    if 'TDR5006B11724' in data.columns:
        if 'TDR5006B11824' in data.columns:
            logger.warning("Both correct and incorrect column names exist for TDR5006B11824. Merging data.")
            data['TDR5006B11824'] = data['TDR5006B11824'].fillna(data['TDR5006B11724'])
        else:
            data['TDR5006B11824'] = data['TDR5006B11724']
        data.drop('TDR5006B11724', axis=1, inplace=True)
        logger.info("Corrected misspelled column name from TDR5006B11724 to TDR5006B11824")

    if 'TDR5026A23824' in data.columns:
        if 'TDR5026A23024' in data.columns:
            logger.warning("Both correct and incorrect column names exist for TDR5026A23024. Merging data.")
            data['TDR5026A23024'] = data['TDR5026A23024'].fillna(data['TDR5026A23824'])
        else:
            data['TDR5026A23024'] = data['TDR5026A23824']
        data.drop('TDR5026A23824', axis=1, inplace=True)
        logger.info("Corrected misspelled column name from TDR5026A23824 to TDR5026A23024")
    
    logger.info(f"Final columns: {data.columns.tolist()}")
    logger.info(f"Final shape: {data.shape}")
    logger.info(f"TIMESTAMP dtype: {data.index.dtype}")
    logger.info(f"TIMESTAMP null count: {data.index.isnull().sum()}")
    
    return data

def insert_or_update_data(client, table_id, df, is_actual=True):
    logger.info(f"Preparing to {'insert' if is_actual else 'update'} data in {table_id}")
    logger.info(f"DataFrame columns: {df.columns.tolist()}")
    logger.info(f"DataFrame shape: {df.shape}")

    # Ensure TIMESTAMP is in UTC
    df.index = df.index.tz_convert('UTC')

    # Add is_actual column
    df['is_actual'] = is_actual

    # Get current schema
    table = client.get_table(table_id)
    current_schema = {field.name: field.field_type for field in table.schema}
    logger.info(f"Current schema for table {table_id}: {current_schema}")

    # Prepare data for upload
    data_to_upload = df.reset_index()  # Reset index to make TIMESTAMP a column
    
    # Only include columns that are in both the schema and the dataframe
    columns_to_upload = [col for col in current_schema if col in data_to_upload.columns]
    data_to_upload = data_to_upload[columns_to_upload]

    # Fill NaN values with None for BigQuery compatibility
    data_to_upload = data_to_upload.where(pd.notnull(data_to_upload), None)

    # Sort the dataframe by TIMESTAMP
    data_to_upload = data_to_upload.sort_values('TIMESTAMP')

    # Create a new schema only including the columns we're uploading
    upload_schema = [field for field in table.schema if field.name in columns_to_upload]

    # Upload data
    job_config = bigquery.LoadJobConfig(
        schema=upload_schema,
        write_disposition="WRITE_APPEND"
    )
    try:
        job = client.load_table_from_dataframe(data_to_upload, table_id, job_config=job_config)
        job.result()  # Wait for the job to complete
        logger.info(f"Successfully loaded {len(data_to_upload)} rows into {table_id}")
    except Exception as e:
        logger.error(f"Error uploading data to {table_id}: {str(e)}")
        raise

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
        full_table_id = f"{client.project}.{dataset_id}.plot_{plot_number}"
        
        # Select relevant columns for this group
        columns_to_upload = sensors  # We don't need to include 'TIMESTAMP' as it's the index
        df_to_upload = df[df.columns.intersection(columns_to_upload)].copy()
        
        if not df_to_upload.empty:
            try:
                insert_or_update_data(client, full_table_id, df_to_upload, is_actual)
            except Exception as e:
                logger.error(f"Failed to upload data for plot {plot_number} to {full_table_id}: {str(e)}")
        else:
            logger.info(f"No data to upload for plot {plot_number}")

def main():
    # Load the sensor mapping
    sensor_mapping = load_sensor_mapping()
    
    dat_files = [
        r"C:\Campbellsci\PC400\nodeC_NodeC.dat",
        r"C:\Campbellsci\PC400\nodeB_NodeB.dat",
        r"C:\Campbellsci\PC400\nodeA_NodeA.dat"
    ]

    for dat_file in dat_files:
        logger.info(f"Processing file: {dat_file}")
        df = parse_dat_file(dat_file)
        
        # Process and upload data for each sensor in the dataframe
        process_and_upload_data(df, sensor_mapping, is_actual=True)

if __name__ == "__main__":
    main()