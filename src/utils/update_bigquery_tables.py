import os
import re   
import pandas as pd
import numpy as np
from google.cloud import bigquery
from google.api_core.exceptions import NotFound
from logger import get_logger
from bigquery_operations import load_sensor_mapping, create_bigquery_client, insert_data

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

    for col in data.columns:
        if col != "TIMESTAMP":
            data[col] = pd.to_numeric(data[col], errors='coerce')
    
    data["TIMESTAMP"] = pd.to_datetime(data["TIMESTAMP"], errors="coerce")
    logger.info(f"Number of rows before dropping NaT timestamps: {len(data)}")
    data = data[~data["TIMESTAMP"].isna()]
    logger.info(f"Number of rows after dropping NaT timestamps: {len(data)}")
    
    data = data.set_index("TIMESTAMP")

    data.index = data.index.tz_localize("America/Chicago")
    
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

def get_dat_files(folder_path, crop_type):
    if crop_type == 'corn':
        patterns = [
            r'nodeA.*\.dat',
            r'nodeB.*\.dat',
            r'nodeC.*\.dat'
        ]
    elif crop_type == 'soybean':
        patterns = [
            r'SoyNodeA.*_NodeA\.dat',
            r'SoyNodeB.*_NodeB\.dat',
            r'SoyNodeC.*_NodeC\.dat'
        ]
    
    dat_files = []
    for file in os.listdir(folder_path):
        for pattern in patterns:
            if re.match(pattern, file, re.IGNORECASE):
                dat_files.append(os.path.join(folder_path, file))
                break
    return dat_files

def process_folder(folder_path, sensor_mapping, crop_type):
    dat_files = get_dat_files(folder_path, crop_type)

    for dat_file in dat_files:
        if os.path.exists(dat_file):
            logger.info(f"Processing file: {dat_file}")
            df = parse_dat_file(dat_file)
            crop_specific_mapping = [sensor for sensor in sensor_mapping if sensor['field'] == f'LINEAR_{crop_type.upper()}']
            process_and_upload_data(df, crop_specific_mapping, crop_type=crop_type)
        else:
            logger.warning(f"File not found: {dat_file}")

def process_and_upload_data(df, sensor_mapping, crop_type='corn'):
    client = create_bigquery_client()
    
    # Reset the index to make 'TIMESTAMP' a regular column
    df = df.reset_index()
    
    sensor_groups = {}
    for sensor in sensor_mapping:
        key = (sensor['treatment'], sensor['plot_number'], sensor['field'])
        if key not in sensor_groups:
            sensor_groups[key] = []
        sensor_groups[key].append(sensor['sensor_id'])

    for (treatment, plot_number, field), sensors in sensor_groups.items():
        table_id = f"{field}_trt{treatment}.plot_{plot_number}"
        dataset_id = f"{field}_trt{treatment}"
        full_table_id = f"{client.project}.{dataset_id}.plot_{plot_number}"
        
        columns_to_upload = ['TIMESTAMP'] + sensors
        df_to_upload = df[df.columns.intersection(columns_to_upload)].copy()
        
        if not df_to_upload.empty:
            try:
                # Check if the table exists, if not, create it
                try:
                    client.get_table(full_table_id)
                except NotFound:
                    logger.info(f"Table {full_table_id} not found. Creating it.")
                    create_table(client, full_table_id, df_to_upload.columns)

                insert_data(client, full_table_id, df_to_upload)
            except Exception as e:
                logger.error(f"Failed to upload data for {field} plot {plot_number} to {full_table_id}: {str(e)}")
        else:
            logger.info(f"No data to upload for {field} plot {plot_number}")

def create_table(client, full_table_id, columns):
    schema = [
        bigquery.SchemaField("TIMESTAMP", "TIMESTAMP", mode="REQUIRED"),
    ]
    for column in columns:
        if column != "TIMESTAMP":
            schema.append(bigquery.SchemaField(column, "FLOAT"))

    table = bigquery.Table(full_table_id, schema=schema)
    client.create_table(table)
    logger.info(f"Created table {full_table_id}")

def main(corn_folders, soybean_folders):
    sensor_mapping = load_sensor_mapping()
    
    logger.info("Processing corn data")
    for folder in corn_folders:
        logger.info(f"Processing corn folder: {folder}")
        process_folder(folder, sensor_mapping, crop_type='corn')
    
    logger.info("Processing soybean data")
    for folder in soybean_folders:
        logger.info(f"Processing soybean folder: {folder}")
        process_folder(folder, sensor_mapping, crop_type='soybean')

if __name__ == "__main__":
    corn_folders = [
        r"C:\Users\bnsoh2\OneDrive - University of Nebraska-Lincoln\Projects\Students\Bryan Nsoh\Data\2024_data_corn_lnr\07-03-2024",
        r"c:\Users\bnsoh2\OneDrive - University of Nebraska-Lincoln\Projects\Students\Bryan Nsoh\Data\2024_data_corn_lnr\07-08-2024-discontinuous",
        r"C:\Users\bnsoh2\OneDrive - University of Nebraska-Lincoln\Projects\Students\Bryan Nsoh\Data\2024_data_corn_lnr\07-14-2024-discont-nodeC only",
        r"C:\Users\bnsoh2\OneDrive - University of Nebraska-Lincoln\Projects\Students\Bryan Nsoh\Data\2024_data_corn_lnr\07-15-2024-discont-unsure",
        r"C:\Users\bnsoh2\OneDrive - University of Nebraska-Lincoln\Projects\Students\Bryan Nsoh\Data\2024_data_corn_lnr\07-19-2024"
    ]
    soybean_folders = [
        r"C:\Users\bnsoh2\OneDrive - University of Nebraska-Lincoln\Projects\Students\Bryan Nsoh\Data\Soybean Lnr\07-15-24",
        r"C:\Users\bnsoh2\OneDrive - University of Nebraska-Lincoln\Projects\Students\Bryan Nsoh\Data\Soybean Lnr\07-19-2024"
    ]
    main(corn_folders, soybean_folders)