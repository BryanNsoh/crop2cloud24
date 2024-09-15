import sqlite3
import pandas as pd
import logging
from ..config import DB_NAME

logger = logging.getLogger(__name__)

def create_plot_table(conn, crop_type, treatment, plot_number):
    conn.execute(f"""
    CREATE TABLE IF NOT EXISTS {crop_type}_{treatment}_plot_{plot_number} (
        TIMESTAMP TEXT PRIMARY KEY,
        TDR_{plot_number}_{treatment}0624 REAL,
        SAP_{plot_number}_{treatment}xx24 REAL,
        TDR_{plot_number}_{treatment}3024 REAL,
        IRT_{plot_number}_{treatment}xx24 REAL,
        TDR_{plot_number}_{treatment}1824 REAL,
        DEN_{plot_number}_{treatment}xx24 REAL,
        TDR_{plot_number}_{treatment}4224 REAL,
        cwsi REAL,
        et REAL,
        swsi REAL,
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