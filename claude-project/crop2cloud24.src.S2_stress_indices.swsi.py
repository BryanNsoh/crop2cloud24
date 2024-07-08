import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import logging
import sys
import time
import random

"""
This script calculates the Soil Water Stress Index (SWSI) for agricultural plots.

Data Sources:
1. Soil properties: Obtained from Web Soil Survey, specific to the experimental site.
   Includes saturation, field capacity, permanent wilting point, and bulk density for different soil layers.
2. Plot data: Retrieved from a SQLite database ('mpc_data.db') containing time series of soil moisture measurements.
3. Management Allowed Depletion (MAD): Set to 0.45 (45%) based on the recommendation by Panda et al. (2004) 
   for maize grown in sandy loam soils in sub-tropical regions.

Calculations:
1. Volumetric Water Content (VWC): Measured directly by soil moisture sensors at different depths.
2. Weighted averages: Calculated for field capacity (fc) and permanent wilting point (pwp) across soil layers.
3. Available Water Capacity (AWC): Difference between weighted fc and weighted pwp.
4. Threshold VWC (VWCt): Calculated as weighted_fc - MAD * AWC.
5. SWSI: Calculated as (VWCt - avg_vwc) / (VWCt - weighted_pwp) when avg_vwc < VWCt, otherwise 0.

The SWSI calculation methodology is based on the paper:
Panda, R.K., Behera, S.K., Kashyap, P.S., 2004. Effective management of irrigation water for maize under 
stressed conditions. Agricultural Water Management, 66(3), 181-203.
https://doi.org/10.1016/j.agwat.2003.12.001

The paper recommends scheduling irrigation at 45% MAD of available soil water during non-critical growth 
stages for optimal yield, water use efficiency, and net return for maize in sandy loam soils in sub-tropical regions.
"""

# Configuration
DB_PATH = 'mpc_data.db'
PLOTS = ['plot_5006', 'plot_5010', 'plot_5023']

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Soil properties
# Obtained from Web Soil Survey
SOIL_LAYERS = [
    {"depth": (0, 12), "saturation": 0.528, "fc": 0.277, "pwp": 0.123, "bulk_density": 1.25},
    {"depth": (12, 18), "saturation": 0.509, "fc": 0.282, "pwp": 0.129, "bulk_density": 1.3},
    {"depth": (18, 79), "saturation": 0.472, "fc": 0.243, "pwp": 0.085, "bulk_density": 1.4}
]

def get_db_connection():
    return sqlite3.connect(DB_PATH)

def get_plot_data(conn, plot_number):
    tdr_columns = {
        'plot_5006': ['TDR5006B10624', 'TDR5006B11824', 'TDR5006B13024'],
        'plot_5010': ['TDR5010C10624', 'TDR5010C11824', 'TDR5010C13024'],
        'plot_5023': ['TDR5023A10624', 'TDR5023A11824', 'TDR5023A13024']
    }
    
    columns = ", ".join(tdr_columns[plot_number])
    query = f"""
    SELECT TIMESTAMP, {columns}
    FROM {plot_number}
    WHERE TIMESTAMP IS NOT NULL
    ORDER BY TIMESTAMP
    """
    df = pd.read_sql_query(query, conn, parse_dates=['TIMESTAMP'])
    
    # Handle NaT values in TIMESTAMP
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], utc=True, errors='coerce')
    df = df.dropna(subset=['TIMESTAMP'])
    
    # Convert VWC values from percent to decimal and handle NaN values
    for col in tdr_columns[plot_number]:
        df[col] = pd.to_numeric(df[col], errors='coerce') / 100.0
    
    # Drop rows where all TDR columns are NaN
    df = df.dropna(subset=tdr_columns[plot_number], how='all')
    
    logger.info(f"{plot_number}: Retrieved {len(df)} rows. Date range: {df['TIMESTAMP'].min()} to {df['TIMESTAMP'].max()}")
    
    # Log VWC statistics for each depth
    for col in tdr_columns[plot_number]:
        valid_vwc = df[col].dropna()
        logger.info(f"{plot_number} - {col} VWC stats: Min: {valid_vwc.min():.4f}, Max: {valid_vwc.max():.4f}, Median: {valid_vwc.median():.4f}")
    
    return df

def calculate_swsi(vwc_values, plot_number):
    """Calculate SWSI for a set of VWC values."""
    MAD = 0.45  # management allowed depletion, set to 45% as per Panda et al. (2004)
    
    valid_vwc = [vwc for vwc in vwc_values if pd.notna(vwc)]
    
    if len(valid_vwc) < 3:
        return None, None, None, None, None, None, f"Insufficient valid VWC values: {len(valid_vwc)}"
    
    avg_vwc = np.mean(valid_vwc)

    # Calculate weighted average of soil properties based on sensor depths
    total_depth = sum(layer['depth'][1] - layer['depth'][0] for layer in SOIL_LAYERS)
    weighted_fc = sum(layer['fc'] * (layer['depth'][1] - layer['depth'][0]) / total_depth for layer in SOIL_LAYERS)
    weighted_pwp = sum(layer['pwp'] * (layer['depth'][1] - layer['depth'][0]) / total_depth for layer in SOIL_LAYERS)
    
    AWC = weighted_fc - weighted_pwp  # Available water capacity of soil
    VWCt = weighted_fc - MAD * AWC  # threshold for triggering irrigation

    if avg_vwc < VWCt:
        swsi = (VWCt - avg_vwc) / (VWCt - weighted_pwp)
        return swsi, avg_vwc, weighted_fc, weighted_pwp, VWCt, AWC, None
    else:
        return 0, avg_vwc, weighted_fc, weighted_pwp, VWCt, AWC, "VWC above threshold"

def compute_swsi(plot_number, df):
    logger.info(f"Computing SWSI for {plot_number}")
    swsi_values = []
    all_swsi = []
    error_counts = {"Insufficient VWC": 0, "VWC above threshold": 0, "Other": 0}
    
    # Print VWCt and AWC once for this plot
    vwc_values = df.iloc[0][df.columns[df.columns.str.startswith('TDR')]].tolist()
    _, _, weighted_fc, weighted_pwp, VWCt, AWC, _ = calculate_swsi(vwc_values, plot_number)
    logger.info(f"{plot_number} - VWCt: {VWCt:.4f}, AWC: {AWC:.4f}, Weighted FC: {weighted_fc:.4f}, Weighted PWP: {weighted_pwp:.4f}")
    
    for _, row in df.iterrows():
        vwc_values = [row[col] for col in df.columns if col.startswith('TDR')]
        swsi, avg_vwc, _, _, _, _, error_reason = calculate_swsi(vwc_values, plot_number)
        
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
            'swsi': swsi,
            'is_actual': True
        })
    
    swsi_df = pd.DataFrame(swsi_values)
    
    # Log summary statistics
    logger.info(f"SWSI Summary for {plot_number}:")
    logger.info(f"  Total timestamps processed: {len(df)}")
    logger.info(f"  SWSI calculated for: {len(all_swsi)} timestamps")
    logger.info(f"  Failed calculations:")
    for reason, count in error_counts.items():
        logger.info(f"    {reason}: {count}")
    
    if all_swsi:
        logger.info(f"  SWSI statistics: Min: {min(all_swsi):.4f}, Max: {max(all_swsi):.4f}, Avg: {np.mean(all_swsi):.4f}, Median: {np.median(all_swsi):.4f}")
    else:
        logger.warning(f"  No valid SWSI values calculated for {plot_number}")
    
    return swsi_df

def update_swsi_in_plot_tables(conn, plot_number, swsi_df):
    logger.info(f"Updating SWSI for {plot_number}")
    cursor = conn.cursor()
    
    # Update SWSI values
    rows_updated = 0
    for _, row in swsi_df.iterrows():
        swsi_timestamp = row['TIMESTAMP']
        if pd.isna(swsi_timestamp):
            continue
        
        start_time = (swsi_timestamp - timedelta(minutes=30)).strftime('%Y-%m-%d %H:%M:%S')
        end_time = (swsi_timestamp + timedelta(minutes=30)).strftime('%Y-%m-%d %H:%M:%S')
        
        # Use a 1-hour window to match timestamps
        cursor.execute(f"""
        UPDATE {plot_number}
        SET swsi = ?, is_actual = ?
        WHERE TIMESTAMP BETWEEN ? AND ?
        """, (row['swsi'], row['is_actual'], start_time, end_time))
        rows_updated += cursor.rowcount
    
    conn.commit()
    logger.info(f"Updated {rows_updated} rows in {plot_number}")
    
    # Log new SWSI values
    cursor.execute(f"SELECT MIN(swsi), MAX(swsi), AVG(swsi) FROM {plot_number} WHERE swsi IS NOT NULL")
    min_swsi, max_swsi, avg_swsi = cursor.fetchone()
    if min_swsi is not None and max_swsi is not None and avg_swsi is not None:
        logger.info(f"New SWSI values in {plot_number}: Min: {min_swsi:.4f}, Max: {max_swsi:.4f}, Avg: {avg_swsi:.4f}")
    else:
        logger.warning(f"No valid SWSI values found in {plot_number} after update")

def main():
    start_time = time.time()
    logger.info("Starting SWSI computation")
    
    conn = get_db_connection()
    
    for plot in PLOTS:
        try:
            df = get_plot_data(conn, plot)
            if df.empty:
                logger.warning(f"No data available for {plot}")
                continue
            
            swsi_df = compute_swsi(plot, df)
            update_swsi_in_plot_tables(conn, plot, swsi_df)
        except Exception as e:
            logger.error(f"Error processing {plot}: {str(e)}")
    
    conn.close()
    
    end_time = time.time()
    duration = end_time - start_time
    logger.info(f"SWSI computation completed. Total execution time: {duration:.2f} seconds")

if __name__ == "__main__":
    main()