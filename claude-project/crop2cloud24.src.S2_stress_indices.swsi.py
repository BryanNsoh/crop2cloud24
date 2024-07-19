import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import logging
import time
from dotenv import load_dotenv
from crop2cloud24.src.utils import generate_plots

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
        return f"{datetime.now(pytz.timezone('America/Chicago')).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} CST - {record.levelname} - {record.message}"

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

class SWSICalculator:
    """
    This class calculates the Soil Water Stress Index (SWSI) for agricultural plots.

    Data Sources:
    1. Soil properties: Obtained from Web Soil Survey, specific to the experimental site.
       Includes field capacity and permanent wilting point for different soil types.
    2. Plot data: Retrieved from a SQLite database containing time series of soil moisture measurements.
    3. Management Allowed Depletion (MAD): Set to 0.45 (45%) based on the recommendation by Panda et al. (2004) 
       for maize grown in sandy loam soils in sub-tropical regions.

    Calculations:
    1. Volumetric Water Content (VWC): Measured directly by soil moisture sensors at different depths.
    2. Weighted averages: Calculated for field capacity (fc) and permanent wilting point (pwp) based on soil composition.
    3. Available Water Capacity (AWC): Difference between fc and pwp.
    4. Threshold VWC (VWCt): Calculated as fc - MAD * AWC.
    5. SWSI: Calculated as (VWCt - avg_vwc) / (VWCt - pwp) when avg_vwc < VWCt, otherwise 0.

    The SWSI calculation methodology is based on the paper:
    Panda, R.K., Behera, S.K., Kashyap, P.S., 2004. Effective management of irrigation water for maize under 
    stressed conditions. Agricultural Water Management, 66(3), 181-203.
    https://doi.org/10.1016/j.agwat.2003.12.001

    The paper recommends scheduling irrigation at 45% MAD of available soil water during non-critical growth 
    stages for optimal yield, water use efficiency, and net return for maize in sandy loam soils in sub-tropical regions.
    """

    def __init__(self, db_path):
        self.data_manager = DataManager(db_path)
        self.logger = logging.getLogger(__name__)
        
        # Soil data from Web Soil Survey (WSS)
        self.soil_data = {
            '8815': {'fc': 0.269, 'pwp': 0.115},
            '8816': {'fc': 0.279, 'pwp': 0.126},
            '8869': {'fc': 0.291, 'pwp': 0.143}
        }
        
        # Plot-specific soil compositions
        self.plot_soil_composition = {
            'plot_5006': {'8816': 1.0},
            'plot_5023': {'8816': 0.1, '8815': 0.9},
            'plot_5010': {'8816': 1.0}
        }
        
        # Management Allowed Depletion (MAD)
        self.MAD = 0.45
        
        # TDR columns for each plot
        self.tdr_columns = {
            'plot_5006': ['TDR5006B10624', 'TDR5006B11824', 'TDR5006B13024'],
            'plot_5010': ['TDR5010C10624', 'TDR5010C11824', 'TDR5010C13024'],
            'plot_5023': ['TDR5023A10624', 'TDR5023A11824', 'TDR5023A13024']
        }

    def calculate_plot_soil_properties(self, plot_number):
        """
        Calculate soil properties for a specific plot based on its soil composition.

        Args:
            plot_number (str): The plot number to calculate properties for.

        Returns:
            tuple: (field capacity, permanent wilting point, available water capacity, volumetric water content threshold)
        """
        composition = self.plot_soil_composition[plot_number]
        fc = sum(self.soil_data[soil_type]['fc'] * ratio for soil_type, ratio in composition.items())
        pwp = sum(self.soil_data[soil_type]['pwp'] * ratio for soil_type, ratio in composition.items())
        awc = fc - pwp
        vwct = fc - self.MAD * awc
        return fc, pwp, awc, vwct

    def get_plot_data(self, plot_number):
        """
        Retrieve and preprocess data for a specific plot.

        Args:
            plot_number (str): The plot number to retrieve data for.

        Returns:
            pandas.DataFrame: Preprocessed data for the specified plot.
        """
        columns = ['TIMESTAMP'] + self.tdr_columns[plot_number]
        df = self.data_manager.load_data(plot_number, columns=columns)
        
        for col in self.tdr_columns[plot_number]:
            df[col] = pd.to_numeric(df[col], errors='coerce') / 100.0
        
        df = df.dropna(subset=self.tdr_columns[plot_number], how='all')
        
        self.logger.info(f"{plot_number}: Retrieved {len(df)} rows. Date range: {df['TIMESTAMP'].min()} to {df['TIMESTAMP'].max()}")
        
        return df

    def calculate_swsi(self, vwc_values, plot_number):
        """
        Calculate SWSI for a set of VWC values.

        Args:
            vwc_values (list): List of volumetric water content values.
            plot_number (str): The plot number for which SWSI is being calculated.

        Returns:
            tuple: (SWSI value, average VWC, error message if any)
        """
        valid_vwc = [vwc for vwc in vwc_values if pd.notna(vwc)]
        
        if len(valid_vwc) < 3:
            return None, None, f"Insufficient valid VWC values: {len(valid_vwc)}"
        
        avg_vwc = np.mean(valid_vwc)
        fc, pwp, awc, vwct = self.calculate_plot_soil_properties(plot_number)
        
        if avg_vwc < vwct:
            swsi = (vwct - avg_vwc) / (vwct - pwp)
            return swsi, avg_vwc, None
        else:
            return 0, avg_vwc, "VWC above threshold"

    def compute_swsi_for_plot(self, plot_number):
        """
        Compute SWSI for all timestamps of a specific plot.

        Args:
            plot_number (str): The plot number to compute SWSI for.

        Returns:
            pandas.DataFrame: DataFrame containing TIMESTAMP and calculated SWSI values.
        """
        df = self.get_plot_data(plot_number)
        if df.empty:
            self.logger.warning(f"No data available for {plot_number}")
            return pd.DataFrame()

        swsi_values = []
        all_swsi = []
        error_counts = {"Insufficient VWC": 0, "VWC above threshold": 0, "Other": 0}
        
        fc, pwp, awc, vwct = self.calculate_plot_soil_properties(plot_number)
        self.logger.info(f"{plot_number} - VWCt: {vwct:.4f}, AWC: {awc:.4f}, FC: {fc:.4f}, PWP: {pwp:.4f}")
        
        for _, row in df.iterrows():
            vwc_values = [row[col] for col in self.tdr_columns[plot_number]]
            swsi, avg_vwc, error_reason = self.calculate_swsi(vwc_values, plot_number)
            
            if swsi is not None:
                all_swsi.append(swsi)
            elif error_reason:
                if "Insufficient" in error_reason:
                    error_counts["Insufficient VWC"] += 1
                elif "VWC above threshold" in error_reason:
                    error_counts["VWC above threshold"] += 1
                else:
                    error_counts["Other"] += 1
            
            swsi_values.append({
                'TIMESTAMP': row['TIMESTAMP'],
                'swsi': swsi if swsi is not None else 0  # Replace None with 0
            })
        
        swsi_df = pd.DataFrame(swsi_values)
        
        self.log_swsi_summary(plot_number, df, all_swsi, error_counts)
        
        return swsi_df

    def log_swsi_summary(self, plot_number, df, all_swsi, error_counts):
        """
        Log summary statistics for SWSI calculations.

        Args:
            plot_number (str): The plot number being processed.
            df (pandas.DataFrame): The original data frame for the plot.
            all_swsi (list): List of all valid SWSI values calculated.
            error_counts (dict): Dictionary containing counts of different error types.
        """
        self.logger.info(f"SWSI Summary for {plot_number}:")
        self.logger.info(f"  Total timestamps processed: {len(df)}")
        self.logger.info(f"  SWSI calculated for: {len(all_swsi)} timestamps")
        self.logger.info(f"  Failed calculations:")
        for reason, count in error_counts.items():
            self.logger.info(f"    {reason}: {count}")
        
        if all_swsi:
            self.logger.info(f"  SWSI statistics: Min: {min(all_swsi):.4f}, Max: {max(all_swsi):.4f}, Avg: {np.mean(all_swsi):.4f}, Median: {np.median(all_swsi):.4f}")
        else:
            self.logger.warning(f"  No valid SWSI values calculated for {plot_number}")

    def update_swsi_in_database(self, plot_number, swsi_df):
        """
        Update the calculated SWSI values in the database for a specific plot.

        Args:
            plot_number (str): The plot number to update SWSI values for.
            swsi_df (pandas.DataFrame): DataFrame containing TIMESTAMP and SWSI values.
        """
        self.logger.info(f"Updating SWSI for {plot_number}")
        self.data_manager.update_database(plot_number, swsi_df, 'swsi')
        
        updated_data = self.data_manager.load_data(plot_number, columns=['TIMESTAMP', 'swsi'])
        valid_swsi = updated_data['swsi'].dropna()
        if not valid_swsi.empty:
            self.logger.info(f"New SWSI values in {plot_number}: Min: {valid_swsi.min():.4f}, Max: {valid_swsi.max():.4f}, Avg: {valid_swsi.mean():.4f}")
        else:
            self.logger.warning(f"No valid SWSI values found in {plot_number} after update")

    def run(self):
        """
        Run the SWSI calculation process for all plots.
        """
        for plot in self.plot_soil_composition.keys():
            try:
                swsi_df = self.compute_swsi_for_plot(plot)
                if not swsi_df.empty:
                    self.update_swsi_in_database(plot, swsi_df)
            except Exception as e:
                self.logger.error(f"Error processing {plot}: {str(e)}")

def main():
    logger.info(f"Using database: {DB_PATH}")
    swsi_calculator = SWSICalculator(DB_PATH)
    swsi_calculator.run()

if __name__ == "__main__":
    main()