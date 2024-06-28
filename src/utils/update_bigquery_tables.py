import os
import pandas as pd
import numpy as np
from google.cloud import bigquery
from google.api_core.exceptions import NotFound
from logger import get_logger
from bigquery_operations import load_sensor_mapping, process_and_upload_data, create_bigquery_client

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
    data_hourly = data.resample("h").mean()

    data_hourly.reset_index(inplace=True)
    # Convert TIMESTAMP to datetime64[ns] explicitly
    data_hourly["TIMESTAMP"] = pd.to_datetime(data_hourly["TIMESTAMP"], utc=True).dt.tz_convert('America/Chicago').dt.tz_localize(None)
    
    logger.info(f"Final columns: {data_hourly.columns.tolist()}")
    logger.info(f"Final shape: {data_hourly.shape}")
    logger.info(f"TIMESTAMP dtype: {data_hourly['TIMESTAMP'].dtype}")
    logger.info(f"TIMESTAMP null count: {data_hourly['TIMESTAMP'].isnull().sum()}")
    
    return data_hourly

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
        process_and_upload_data(df, sensor_mapping)

if __name__ == "__main__":
    main()