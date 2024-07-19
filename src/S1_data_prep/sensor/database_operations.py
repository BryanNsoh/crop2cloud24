import sqlite3
import pandas as pd
import logging
from ..config import DB_NAME

logger = logging.getLogger(__name__)

def create_plot_table(conn, crop_type, treatment, plot_number):
    conn.execute(f"""
    CREATE TABLE IF NOT EXISTS {crop_type}_{treatment}_plot_{plot_number} (
        TIMESTAMP TEXT PRIMARY KEY,
        is_actual INTEGER,
        prediction_timestamp TEXT,
        applied_irrigation REAL,
        TDR_{plot_number}_10624 REAL,
        TDR_{plot_number}_10624_pred REAL,
        SAP_{plot_number}_1xx24 REAL,
        SAP_{plot_number}_1xx24_pred REAL,
        TDR_{plot_number}_13024 REAL,
        TDR_{plot_number}_13024_pred REAL,
        IRT_{plot_number}_1xx24 REAL,
        IRT_{plot_number}_1xx24_pred REAL,
        TDR_{plot_number}_11824 REAL,
        TDR_{plot_number}_11824_pred REAL,
        DEN_{plot_number}_1xx24 REAL,
        DEN_{plot_number}_1xx24_pred REAL,
        TDR_{plot_number}_14224 REAL,
        TDR_{plot_number}_14224_pred REAL,
        cwsi_th2 REAL,
        cwsi_th2_pred REAL,
        et REAL,
        et_pred REAL,
        swsi REAL,
        swsi_pred REAL
    )
    """)
    logger.info(f"Created {crop_type}_{treatment}_plot_{plot_number} table")

def store_plot_data(plot_data):
    logger.info("Storing plot data")
    conn = sqlite3.connect(DB_NAME)
    
    for crop_type, treatments in plot_data.items():
        for treatment, plots in treatments.items():
            for plot_number, df in plots.items():
                logger.debug(f"Processing {crop_type}_{treatment}_plot_{plot_number} with data type: {type(df)}")
                if isinstance(df, pd.DataFrame):
                    create_plot_table(conn, crop_type, treatment, plot_number)
                    df.to_sql(f'{crop_type}_{treatment}_plot_{plot_number}', conn, if_exists='replace', index=False)
                    logger.info(f"Stored {len(df)} records for {crop_type} {treatment} plot {plot_number}")
                else:
                    logger.error(f"Expected DataFrame but got {type(df)} for {crop_type}_{treatment}_plot_{plot_number}")
    
    conn.close()
    logger.info("Finished storing plot data")