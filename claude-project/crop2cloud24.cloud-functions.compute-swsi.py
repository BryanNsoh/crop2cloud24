import os
from google.cloud import bigquery
from datetime import datetime, timedelta
import pytz
import logging
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# BigQuery details
PROJECT_ID = "crop2cloud24"
DATASET_ID = "LINEAR_CORN_trt1"

def calculate_swsi(vwc_values):
    """Calculate SWSI for a set of VWC values."""
    MAD = 0.5  # management allowed depletion
    VWC_WP = 0.11  # volumetric water content at wilting point
    VWC_FC = 0.29  # volumetric water content at field capacity
    AWC = VWC_FC - VWC_WP  # Available water capacity of soil
    VWC_MAD = VWC_FC - MAD * AWC  # threshold for triggering irrigation

    valid_vwc = [vwc for vwc in vwc_values if pd.notna(vwc)]
    
    if len(valid_vwc) > 2:
        avg_vwc = np.mean(valid_vwc) / 100  # Convert from percentage to fraction
        if avg_vwc < VWC_MAD:
            return abs(avg_vwc - VWC_MAD) / (VWC_MAD - VWC_WP)
    return None

def get_plot_data(client, plot_number, start_time, end_time):
    tdr_columns = {
        5006: ["TDR5006B10624", "TDR5006B11824", "TDR5006B13024", "TDR5006B14224"],
        5010: ["TDR5010C10624", "TDR5010C11824", "TDR5010C13024"],
        5023: ["TDR5023A10624", "TDR5023A11824", "TDR5023A13024", "TDR5023A14224"]
    }
    
    columns = ", ".join(tdr_columns[plot_number])
    query = f"""
    SELECT TIMESTAMP, {columns}
    FROM `{PROJECT_ID}.{DATASET_ID}.plot_{plot_number}`
    WHERE TIMESTAMP BETWEEN '{start_time}' AND '{end_time}'
      AND is_actual = TRUE
    ORDER BY TIMESTAMP
    """
    logger.info(f"Executing query: {query}")
    try:
        query_job = client.query(query)
        return list(query_job.result())  # Materialize the results
    except Exception as e:
        logger.error(f"Error querying data for plot {plot_number}: {str(e)}")
        return []

def get_table_schema(client, table_id):
    try:
        table = client.get_table(f"{PROJECT_ID}.{DATASET_ID}.{table_id}")
        return table.schema
    except Exception as e:
        logger.error(f"Error getting schema for table {table_id}: {str(e)}")
        return None

def insert_into_bigquery(client, table_id, data_list):
    if not data_list:
        logger.warning(f"No data to insert into {table_id}")
        return

    table_ref = client.dataset(DATASET_ID).table(table_id)
    
    # Get the existing schema
    schema = get_table_schema(client, table_id)
    if not schema:
        logger.error(f"Unable to get schema for table {table_id}")
        return

    job_config = bigquery.LoadJobConfig(schema=schema)
    job_config.write_disposition = bigquery.WriteDisposition.WRITE_APPEND

    try:
        job = client.load_table_from_json(data_list, table_ref, job_config=job_config)
        job.result()  # Wait for the job to complete
        logger.info(f"{len(data_list)} rows have been added successfully to {table_id}.")
    except Exception as e:
        logger.error(f"Error inserting data into {table_id}: {str(e)}")
        raise

def compute_swsi(request):
    try:
        logger.info("Starting SWSI computation function")
        client = bigquery.Client()
        plot_numbers = [5006, 5010, 5023]  # Treatment 1 plot numbers
        
        end_time = datetime.now(pytz.UTC)
        start_time = end_time - timedelta(days=7)  # Process last 7 days of data
        
        for plot_number in plot_numbers:
            logger.info(f"Processing plot {plot_number}")
            rows = get_plot_data(client, plot_number, start_time, end_time)
            
            if not rows:
                logger.warning(f"No data retrieved for plot {plot_number}")
                continue

            swsi_data = []
            for row in rows:
                vwc_values = [row[col] for col in row.keys() if col.startswith(f"TDR{plot_number}")]
                swsi = calculate_swsi(vwc_values)
                if swsi is not None:
                    swsi_data.append({
                        "TIMESTAMP": row["TIMESTAMP"].isoformat(),
                        "swsi": swsi,
                        "is_actual": True  # Set to True as per existing schema
                    })
            
            logger.info(f"Calculated SWSI for {len(swsi_data)} timestamps in plot {plot_number}")
            
            if swsi_data:
                table_id = f"plot_{plot_number}"
                insert_into_bigquery(client, table_id, swsi_data)
            else:
                logger.warning(f"No SWSI data calculated for plot {plot_number}")
        
        logger.info("SWSI computation completed successfully")
        return 'SWSI computation completed successfully', 200
    except Exception as e:
        logger.error(f"Error computing SWSI: {str(e)}", exc_info=True)
        return f'Error computing SWSI: {str(e)}', 500

# For local testing
if __name__ == "__main__":
    compute_swsi(None)