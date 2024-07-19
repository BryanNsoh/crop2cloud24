import sys
import os
import json
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import logging
import time
import traceback
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration
DB_PATH = 'mpc_data.db'
ELEVATION = 876  # meters
LATITUDE = 41.15  # degrees
LONGITUDE = -100.77  # degrees
WIND_HEIGHT = 3  # meters

class CustomFormatter(logging.Formatter):
    def format(self, record):
        return f"{datetime.now(pytz.timezone('America/Chicago')).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} CST - {record.levelname} - {record.getMessage()}"

logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(CustomFormatter())
logger.addHandler(handler)

class DataManager:
    def __init__(self, db_path):
        self.db_path = db_path

    def get_db_connection(self):
        return sqlite3.connect(self.db_path)

    def table_exists(self, table_name):
        conn = self.get_db_connection()
        cursor = conn.cursor()
        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
        result = cursor.fetchone()
        conn.close()
        return result is not None

    def load_data(self, table_name, columns=None, days_back=None):
        conn = self.get_db_connection()
        if columns:
            column_str = ", ".join(columns)
        else:
            column_str = "*"
        
        if days_back is None:
            query = f"SELECT {column_str} FROM {table_name} ORDER BY TIMESTAMP"
        else:
            query = f"""
            SELECT {column_str}
            FROM {table_name}
            WHERE TIMESTAMP >= datetime('now', '-{days_back} days')
            ORDER BY TIMESTAMP
            """
        
        df = pd.read_sql_query(query, conn, parse_dates=['TIMESTAMP'])
        df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], utc=True)
        conn.close()
        
        logger.info(f"Loaded data shape: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")
        return df

    def update_database(self, table_name, df, index_column):
        conn = self.get_db_connection()
        cursor = conn.cursor()
        
        # First, set all rows to NULL for the specific column
        cursor.execute(f"""
        UPDATE {table_name}
        SET {index_column} = NULL
        """)
        
        # Then, update with new values
        for _, row in df.iterrows():
            value = row[index_column] if pd.notna(row[index_column]) else 0  # Replace None with 0
            cursor.execute(f"""
            UPDATE {table_name}
            SET {index_column} = ?
            WHERE TIMESTAMP = ?
            """, (value, row['TIMESTAMP'].strftime('%Y-%m-%d %H:%M:%S')))
        
        conn.commit()
        conn.close()

    def print_db_schema(self):
        conn = self.get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        logger.info(f"Database contains {len(tables)} tables.")
        logger.info(f"Tables: {', '.join([table[0] for table in tables])}")
        conn.close()

class SWSICalculator:
    def __init__(self, db_path, crop_type, plot_mapping):
        self.data_manager = DataManager(db_path)
        self.crop_type = crop_type
        self.plot_mapping = plot_mapping
        self.plot_tables = self.get_plot_tables()
        
        # Soil data from Web Soil Survey (WSS)
        self.soil_data = {
            '8815': {'fc': 0.269, 'pwp': 0.115},
            '8816': {'fc': 0.279, 'pwp': 0.126},
            '8869': {'fc': 0.291, 'pwp': 0.143}
        }
        
        # Calculate average fc and pwp
        self.avg_fc = np.mean([data['fc'] for data in self.soil_data.values()])
        self.avg_pwp = np.mean([data['pwp'] for data in self.soil_data.values()])
        
        # Management Allowed Depletion (MAD)
        self.MAD = 0.45
        
        # TDR columns for each plot
        self.tdr_columns = {
            'LINEAR_CORN_trt1_plot_5006': ['TDR5006B10624', 'TDR5006B11824', 'TDR5006B13024'],
            'LINEAR_CORN_trt1_plot_5010': ['TDR5010C10624', 'TDR5010C11824', 'TDR5010C13024'],
            'LINEAR_CORN_trt1_plot_5023': ['TDR5023A10624', 'TDR5023A11824', 'TDR5023A13024'],
            'LINEAR_CORN_trt2_plot_5003': ['TDR5003C20624', 'TDR5003C21824', 'TDR5003C23024'],
            'LINEAR_CORN_trt2_plot_5012': ['TDR5012B20624', 'TDR5012B21824', 'TDR5012B23024'],
            'LINEAR_CORN_trt2_plot_5026': ['TDR5026A20624', 'TDR5026A21824'],  # Note: Only two TDR columns available
            'LINEAR_CORN_trt3_plot_5001': [],  # No TDR columns available
            'LINEAR_CORN_trt3_plot_5018': [],  # No TDR columns available
            'LINEAR_CORN_trt3_plot_5020': [],  # No TDR columns available
            'LINEAR_CORN_trt4_plot_5007': ['TDR5007B40624', 'TDR5007B41824', 'TDR5007B43024'],
            'LINEAR_CORN_trt4_plot_5009': ['TDR5009C40624', 'TDR5009C41824', 'TDR5009C43024'],
            'LINEAR_CORN_trt4_plot_5027': ['TDR5027A40624', 'TDR5027A41824', 'TDR5027A43024']
        }

    def get_plot_tables(self):
        plot_tables = []
        for treatment, plots in self.plot_mapping[self.crop_type].items():
            trt_number = treatment[3]  # Assuming treatment is 'trt1', 'trt2', etc.
            for plot in plots:
                table_name = f"LINEAR_CORN_{treatment}_plot_{plot}"
                plot_tables.append(table_name)
        return plot_tables

    def calculate_soil_properties(self):
        """
        Calculate soil properties using average fc and pwp values.
        """
        fc = self.avg_fc
        pwp = self.avg_pwp
        awc = fc - pwp
        vwct = fc - self.MAD * awc
        return fc, pwp, awc, vwct

    def get_plot_data(self, plot_number):
        """
        Retrieve and preprocess data for a specific plot.
        """
        columns = ['TIMESTAMP'] + self.tdr_columns[plot_number]
        if not columns:
            logger.warning(f"No TDR columns available for {plot_number}. Skipping.")
            return pd.DataFrame()  # Return an empty DataFrame if no TDR columns

        df = self.data_manager.load_data(plot_number, columns=columns)
        
        for col in self.tdr_columns[plot_number]:
            df[col] = pd.to_numeric(df[col], errors='coerce') / 100.0
        
        df = df.dropna(subset=self.tdr_columns[plot_number], how='all')
        
        logger.info(f"{plot_number}: Retrieved {len(df)} rows. Date range: {df['TIMESTAMP'].min()} to {df['TIMESTAMP'].max()}")
        
        return df

    def calculate_swsi(self, vwc_values):
        """
        Calculate SWSI for a set of VWC values.
        """
        valid_vwc = [vwc for vwc in vwc_values if pd.notna(vwc)]
        
        if len(valid_vwc) < 3:
            return None, None, f"Insufficient valid VWC values: {len(valid_vwc)} out of {len(vwc_values)}"
        
        avg_vwc = np.mean(valid_vwc)
        fc, pwp, awc, vwct = self.calculate_soil_properties()
        
        if avg_vwc < vwct:
            swsi = (vwct - avg_vwc) / (vwct - pwp)
            return swsi, avg_vwc, None
        else:
            return 0, avg_vwc, f"VWC above threshold: {avg_vwc:.4f} > {vwct:.4f}"

    def compute_swsi_for_plot(self, plot_number):
        """
        Compute SWSI for all timestamps of a specific plot.
        """
        logger.info(f"Starting computation for {plot_number}")
        df = self.get_plot_data(plot_number)
        if df.empty:
            logger.warning(f"No data available for {plot_number}")
            return pd.DataFrame()

        logger.info(f"Retrieved {len(df)} rows for {plot_number}. Date range: {df['TIMESTAMP'].min()} to {df['TIMESTAMP'].max()}")

        swsi_values = []
        error_counts = {"Insufficient VWC": 0, "VWC above threshold": 0, "Other": 0}
        
        fc, pwp, awc, vwct = self.calculate_soil_properties()
        logger.info(f"{plot_number} - Soil properties: VWCt: {vwct:.4f}, AWC: {awc:.4f}, FC: {fc:.4f}, PWP: {pwp:.4f}")
        
        for _, row in df.iterrows():
            vwc_values = [row[col] for col in self.tdr_columns[plot_number]]
            swsi, avg_vwc, error_reason = self.calculate_swsi(vwc_values)
            
            if swsi is not None:
                swsi_values.append({'TIMESTAMP': row['TIMESTAMP'], 'swsi': swsi})
            elif error_reason:
                if "Insufficient" in error_reason:
                    error_counts["Insufficient VWC"] += 1
                elif "VWC above threshold" in error_reason:
                    error_counts["VWC above threshold"] += 1
                else:
                    error_counts["Other"] += 1

        swsi_df = pd.DataFrame(swsi_values)
        
        self.log_swsi_summary(plot_number, df, swsi_df, error_counts)
        
        return swsi_df

    def log_swsi_summary(self, plot_number, input_df, swsi_df, error_counts):
        """
        Log summary statistics for SWSI calculations.
        """
        logger.info(f"SWSI Summary for {plot_number}:")
        logger.info(f"  Total timestamps processed: {len(input_df)}")
        logger.info(f"  SWSI calculated for: {len(swsi_df)} timestamps")
        logger.info(f"  Failed calculations:")
        for reason, count in error_counts.items():
            logger.info(f"    {reason}: {count}")
        
        if not swsi_df.empty:
            logger.info(f"  SWSI statistics: Min: {swsi_df['swsi'].min():.4f}, Max: {swsi_df['swsi'].max():.4f}, "
                        f"Avg: {swsi_df['swsi'].mean():.4f}, Median: {swsi_df['swsi'].median():.4f}")
        else:
            logger.warning(f"  No valid SWSI values calculated for {plot_number}")

        logger.info(f"  Input data summary:")
        for col in self.tdr_columns[plot_number]:
            logger.info(f"    {col}: Min: {input_df[col].min():.4f}, Max: {input_df[col].max():.4f}, "
                        f"Avg: {input_df[col].mean():.4f}, Null count: {input_df[col].isnull().sum()}")

    def update_swsi_in_database(self, plot_number, swsi_df):
        """
        Update the calculated SWSI values in the database for a specific plot.
        """
        logger.info(f"Updating SWSI for {plot_number}")
        
        conn = self.data_manager.get_db_connection()
        cursor = conn.cursor()

        for _, row in swsi_df.iterrows():
            swsi_timestamp = row['TIMESTAMP']
            cursor.execute(f"""
            UPDATE `{plot_number}`
            SET swsi = ?
            WHERE TIMESTAMP BETWEEN ? AND ?
            """, (row['swsi'], 
                  (swsi_timestamp - timedelta(minutes=30)).strftime('%Y-%m-%d %H:%M:%S'),
                  (swsi_timestamp + timedelta(minutes=30)).strftime('%Y-%m-%d %H:%M:%S')))
        
        conn.commit()
        conn.close()
        
        updated_data = self.data_manager.load_data(plot_number, columns=['TIMESTAMP', 'swsi'])
        valid_swsi = updated_data['swsi'].dropna()
        if not valid_swsi.empty:
            logger.info(f"New SWSI values in {plot_number}: Min: {valid_swsi.min():.4f}, Max: {valid_swsi.max():.4f}, Avg: {valid_swsi.mean():.4f}")
        else:
            logger.warning(f"No valid SWSI values found in {plot_number} after update")

    def run(self):
        """
        Run the SWSI calculation process for all plots.
        """
        for plot in self.plot_tables:
            if not self.data_manager.table_exists(plot):
                logger.error(f"Table {plot} does not exist in the database. Skipping.")
                continue
            try:
                logger.info(f"Processing plot: {plot}")
                if plot not in self.tdr_columns or not self.tdr_columns[plot]:
                    logger.warning(f"No TDR columns defined for {plot}. Skipping.")
                    continue
                swsi_df = self.compute_swsi_for_plot(plot)
                if swsi_df.empty:
                    logger.warning(f"No SWSI values calculated for {plot}. Check input data.")
                else:
                    self.update_swsi_in_database(plot, swsi_df)
                    logger.info(f"Successfully updated SWSI values for {plot}")
            except Exception as e:
                logger.error(f"Error processing {plot}: {str(e)}", exc_info=True)

def main():
    plot_mapping = {
        "corn": {
            "trt1": [5006, 5010, 5023],
            "trt2": [5003, 5012, 5026],
            "trt3": [5001, 5018, 5020],
            "trt4": [5007, 5009, 5027]
        }
    }
    crop_type = "corn"
    logger.info(f"Using database: {DB_PATH}")
    
    # Print the database schema summary
    data_manager = DataManager(DB_PATH)
    data_manager.print_db_schema()
    
    swsi_calculator = SWSICalculator(DB_PATH, crop_type, plot_mapping)
    swsi_calculator.run()

if __name__ == "__main__":
    main()