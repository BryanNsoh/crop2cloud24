import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import logging
import sys
import time
import requests
import math
import json
from dotenv import load_dotenv
import statistics 
import os
from crop2cloud24.src.utils import generate_plots
import matplotlib.pyplot as plt

# Load environment variables from .env file
load_dotenv()

# Configuration
DB_PATH = 'mpc_data.db'
STEFAN_BOLTZMANN = 5.67e-8
CP = 1005
GRAVITY = 9.81
K = 0.41
CROP_HEIGHT = 1.6
LATITUDE = 41.15
SURFACE_ALBEDO = 0.23

class CustomFormatter(logging.Formatter):
    def format(self, record):
        record.message = record.getMessage()
        return f"{datetime.now(pytz.timezone('America/Chicago')).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} CST - {record.levelname} - {record.message}"

logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(CustomFormatter())
logger.addHandler(handler)

# Use dotenv to get the API key
API_KEY = os.getenv('NDVI_API_KEY')
print(f"Api key is {API_KEY}")
POLYGON_API_URL = "http://api.agromonitoring.com/agro/1.0/polygons"
NDVI_API_URL = "http://api.agromonitoring.com/agro/1.0/ndvi/history"
POLYGON_NAME = "My_Field_Polygon"

def get_or_create_polygon():
    response = requests.get(
        POLYGON_API_URL,
        params={"appid": API_KEY}
    )
    
    if response.status_code == 200:
        polygons = response.json()
        for polygon in polygons:
            if polygon['name'] == POLYGON_NAME:
                logger.info(f"Found existing polygon with id: {polygon['id']}")
                return polygon['id']
    
    coordinates = [
    [-100.774075, 41.090012],  # Northwest corner
    [-100.773341, 41.089999],  # Northeast corner (moved slightly east)
    [-100.773343, 41.088311],  # Southeast corner (moved slightly east)
    [-100.774050, 41.088311],  # Southwest corner
    [-100.774075, 41.090012]   # Closing the polygon
]

    polygon_data = {
        "name": POLYGON_NAME,
        "geo_json": {
            "type": "Feature",
            "properties": {},
            "geometry": {
                "type": "Polygon",
                "coordinates": [coordinates]
            }
        }
    }

    headers = {"Content-Type": "application/json"}

    response = requests.post(
        POLYGON_API_URL,
        params={"appid": API_KEY},
        headers=headers,
        data=json.dumps(polygon_data)
    )

    if response.status_code == 201:
        logger.info("Polygon created successfully")
        return response.json()['id']
    else:
        logger.error(f"Error creating polygon. Status code: {response.status_code}")
        logger.error(response.text)
        return None

def get_latest_ndvi(polygon_id):
    end_date = int(datetime.now().timestamp())
    start_date = end_date - 30 * 24 * 60 * 60

    params = {
        "polyid": polygon_id,
        "start": start_date,
        "end": end_date,
        "appid": API_KEY
    }

    response = requests.get(NDVI_API_URL, params=params)
    if response.status_code != 200:
        logger.error(f"Failed to fetch NDVI data: {response.status_code}")
        return None

    data = response.json()
    if not data:
        logger.warning("No NDVI data available")
        return None

    latest_entry = sorted(data, key=lambda x: x['dt'], reverse=True)[0]
    return latest_entry['data']['mean']

def calculate_lai(ndvi):
    return 0.57 * math.exp(2.33 * ndvi)

def celsius_to_kelvin(temp_celsius):
    return temp_celsius + 273.15

def saturated_vapor_pressure(temperature_celsius):
    return 0.6108 * np.exp(17.27 * temperature_celsius / (temperature_celsius + 237.3))

def vapor_pressure_deficit(temperature_celsius, relative_humidity):
    es = saturated_vapor_pressure(temperature_celsius)
    ea = es * (relative_humidity / 100)
    return es - ea

def net_radiation(solar_radiation, air_temp_celsius, canopy_temp_celsius, surface_albedo=0.23, emissivity_a=0.85, emissivity_c=0.98):
    air_temp_kelvin = celsius_to_kelvin(air_temp_celsius)
    canopy_temp_kelvin = celsius_to_kelvin(canopy_temp_celsius)
    Rns = (1 - surface_albedo) * solar_radiation
    Rnl = emissivity_c * STEFAN_BOLTZMANN * canopy_temp_kelvin**4 - emissivity_a * STEFAN_BOLTZMANN * air_temp_kelvin**4
    return Rns - Rnl

def soil_heat_flux(net_radiation, lai):
    result = net_radiation * 0.1

    return result

def aerodynamic_resistance(wind_speed, measurement_height, zero_plane_displacement, roughness_length):
    return (np.log((measurement_height - zero_plane_displacement) / roughness_length) * 
            np.log((measurement_height - zero_plane_displacement) / (roughness_length * 0.1))) / (K**2 * wind_speed)

def psychrometric_constant(atmospheric_pressure_pa):
    return (CP * atmospheric_pressure_pa) / (0.622 * 2.45e6)

def slope_saturation_vapor_pressure(temperature_celsius):
    return 4098 * saturated_vapor_pressure(temperature_celsius) / (temperature_celsius + 237.3)**2

def convert_wind_speed(u3, crop_height):
    z0 = 0.1 * crop_height
    return u3 * (np.log(2/z0) / np.log(3/z0))

def calculate_cwsi_th1(row, crop_height, lai, latitude, surface_albedo=0.23):
    Ta = row['Ta_2m_Avg']
    RH = row['RH_2m_Avg']
    Rs = row['Solar_2m_Avg']
    u3 = row['WndAveSpd_3m']
    P = row['PresAvg_1pnt5m'] * 100
    Tc = row['canopy_temp']
    
    u2 = convert_wind_speed(u3, crop_height)
    
    if u2 < 0.5 or Ta > 40 or Ta < 0 or RH < 10 or RH > 100:
        logger.warning(f"Extreme weather conditions: u2={u2}, Ta={Ta}, RH={RH}")
        return None
    
    VPD = vapor_pressure_deficit(Ta, RH)
    Rn = net_radiation(Rs, Ta, Tc, surface_albedo)
    G = soil_heat_flux(Rn, lai)
    
    zero_plane_displacement = 0.67 * crop_height
    roughness_length = 0.123 * crop_height
    
    ra = aerodynamic_resistance(u2, 2, zero_plane_displacement, roughness_length)
    γ = psychrometric_constant(P)
    Δ = slope_saturation_vapor_pressure(Ta)
    
    ρ = P / (287.05 * celsius_to_kelvin(Ta))
    
    numerator = (Tc - Ta) - ((ra * (Rn - G)) / (ρ * CP)) + (VPD / γ)
    denominator = ((Δ + γ) * ra * (Rn - G)) / (ρ * CP * γ) + (VPD / γ)
    
    if denominator == 0:
        logger.warning(f"Division by zero encountered: denominator={denominator}")
        return None
    
    cwsi = numerator / denominator
    
    logger.debug(f"CWSI calculation: Ta={Ta}, RH={RH}, u2={u2}, Tc={Tc}, CWSI={cwsi}")
    
    return cwsi

def get_db_connection():
    return sqlite3.connect(DB_PATH)

def get_plot_data(conn, plot_number, irt_column):
    query = f"""
    SELECT TIMESTAMP, {irt_column}
    FROM `LINEAR_CORN_trt1_plot_{plot_number}`
    ORDER BY TIMESTAMP
    """
    df = pd.read_sql_query(query, conn, parse_dates=['TIMESTAMP'])
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], utc=True)
    return df

def get_weather_data(conn, start_time, end_time):
    query = """
    SELECT *
    FROM weather_data
    WHERE TIMESTAMP BETWEEN ? AND ?
    ORDER BY TIMESTAMP
    """
    start_time_str = start_time.strftime('%Y-%m-%d %H:%M:%S')
    end_time_str = end_time.strftime('%Y-%m-%d %H:%M:%S')
    df = pd.read_sql_query(query, conn, params=(start_time_str, end_time_str), parse_dates=['TIMESTAMP'])
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], utc=True)
    return df

def update_cwsi(conn, plot_number, df_cwsi):
    logger.info(f"Updating CWSI for plot {plot_number}")
    
    cursor = conn.cursor()
    
    # Print columns before changes
    cursor.execute(f"PRAGMA table_info(`LINEAR_CORN_trt1_plot_{plot_number}`)")
    columns_before = [column[1] for column in cursor.fetchall()]
    logger.info(f"Columns before update: {columns_before}")
    
    # Check if the column exists, if not, create it
    if 'cwsi_th1' not in columns_before:
        cursor.execute(f"""
        ALTER TABLE `LINEAR_CORN_trt1_plot_{plot_number}`
        ADD COLUMN cwsi_th1 REAL
        """)
        logger.info(f"Added cwsi_th1 column to `LINEAR_CORN_trt1_plot_{plot_number}`")
    
    # Print columns after potential addition
    cursor.execute(f"PRAGMA table_info(`LINEAR_CORN_trt1_plot_{plot_number}`)")
    columns_after = [column[1] for column in cursor.fetchall()]
    logger.info(f"Columns after potential addition: {columns_after}")
    
    # Rename the column in df_cwsi to match the database
    df_cwsi = df_cwsi.rename(columns={'cwsi-th1': 'cwsi_th1'})
    
    # Ensure TIMESTAMP format is consistent
    df_cwsi['TIMESTAMP'] = df_cwsi['TIMESTAMP'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Reset all cwsi_th1 values to NULL
    cursor.execute(f"""
    UPDATE `LINEAR_CORN_trt1_plot_{plot_number}`
    SET cwsi_th1 = NULL
    """)
    logger.info(f"Reset all cwsi_th1 values to NULL")
    
    # Then, update with new values
    updated_rows = 0
    for _, row in df_cwsi.iterrows():
        cursor.execute(f"""
        UPDATE `LINEAR_CORN_trt1_plot_{plot_number}`
        SET cwsi_th1 = ?
        WHERE TIMESTAMP = ?
        """, (row['cwsi_th1'], row['TIMESTAMP']))
        updated_rows += cursor.rowcount
    
    conn.commit()
    
    logger.info(f"Attempted to update {len(df_cwsi)} rows, actually updated {updated_rows} rows")
    
    # Confirm the update
    cursor.execute(f"SELECT COUNT(*) FROM `LINEAR_CORN_trt1_plot_{plot_number}` WHERE cwsi_th1 IS NOT NULL")
    count = cursor.fetchone()[0]
    logger.info(f"Number of rows with CWSI values in DB for plot {plot_number}: {count}")
    
    # Print sample data
    cursor.execute(f"""
    SELECT TIMESTAMP, cwsi_th1 FROM `LINEAR_CORN_trt1_plot_{plot_number}`
    WHERE cwsi_th1 IS NOT NULL
    LIMIT 5
    """)
    sample_data = cursor.fetchall()
    logger.info(f"Sample data after update: {sample_data}")
    
    # Debug: print a few rows from df_cwsi
    logger.info(f"Sample data from df_cwsi:")
    logger.info(df_cwsi.head().to_string())

def plot_cwsi(df_cwsi, plot_number):
    plt.figure(figsize=(10, 6))
    plt.plot(df_cwsi['TIMESTAMP'], df_cwsi['cwsi_th1'], label=f'Plot {plot_number}')
    plt.xlabel('Timestamp')
    plt.ylabel('CWSI')
    plt.title(f'CWSI over Time for Plot {plot_number}')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def update_cwsi_th1(conn, plot_number, df_cwsi):
    logger.info(f"Updating CWSI-TH1 for plot {plot_number}")

    cursor = conn.cursor()

    # Check if the column exists, if not, create it
    cursor.execute(f"PRAGMA table_info(plot_{plot_number})")
    columns = [column[1] for column in cursor.fetchall()]
    if 'cwsi_th1' not in columns:
        cursor.execute(f"ALTER TABLE plot_{plot_number} ADD COLUMN 'cwsi_th1' REAL")
        conn.commit()
        logger.info(f"Added 'cwsi_th1' column to plot_{plot_number} table")

    rows_updated = 0
    for _, row in df_cwsi.iterrows():
        timestamp = row['TIMESTAMP'].tz_convert('UTC')
        start_time = timestamp - timedelta(minutes=30)
        end_time = timestamp + timedelta(minutes=30)
        
        cursor.execute(f"""
        UPDATE plot_{plot_number}
        SET 'cwsi_th1' = ?
        WHERE TIMESTAMP BETWEEN ? AND ?
        """, (row['cwsi_th1'], start_time.strftime('%Y-%m-%d %H:%M:%S'), end_time.strftime('%Y-%m-%d %H:%M:%S')))
        
        if cursor.rowcount > 0:
            rows_updated += cursor.rowcount
        else:
            logger.warning(f"No matching row found for timestamp: {timestamp}")

    conn.commit()
    
    logger.info(f"Successfully updated CWSI-TH1 for plot {plot_number}. Rows updated: {rows_updated}")
    
    # Check for any rows that weren't updated
    cursor.execute(f"""
    SELECT COUNT(*) FROM plot_{plot_number}
    WHERE 'cwsi_th1' IS NULL
    """)
    unupdated_rows = cursor.fetchone()[0]
    logger.info(f"Rows not updated: {unupdated_rows}")

    return rows_updated

def compute_cwsi(plot_number):
    start_time = time.time()
    logger.info(f"Starting CWSI computation for plot {plot_number}")
    
    polygon_id = get_or_create_polygon()
    if polygon_id is None:
        logger.error("Failed to get or create polygon. Aborting CWSI computation.")
        return "CWSI computation aborted due to polygon retrieval/creation failure."
    
    latest_ndvi = get_latest_ndvi(polygon_id)
    if latest_ndvi is None:
        logger.error("Failed to retrieve NDVI data. Aborting CWSI computation.")
        return "CWSI computation aborted due to NDVI data retrieval failure."
    
    LAI = calculate_lai(latest_ndvi)
    logger.info(f"Using NDVI: {latest_ndvi}, Calculated LAI: {LAI}")
    
    conn = get_db_connection()

    irt_column = f'IRT{plot_number}B1xx24' if plot_number == '5006' else f'IRT{plot_number}C1xx24' if plot_number == '5010' else f'IRT{plot_number}A1xx24'
    df = get_plot_data(conn, plot_number, irt_column)
    
    if df.empty:
        logger.info(f"No data for plot {plot_number}")
        conn.close()
        return None
    
    logger.info(f"Processing {len(df)} rows for plot {plot_number}")
    
    # Ensure we're working with hourly data
    df = df.set_index('TIMESTAMP').resample('H').mean().reset_index()
    
    # Convert UTC to CST for filtering
    df['TIMESTAMP_CST'] = df['TIMESTAMP'].dt.tz_convert('America/Chicago')
    df = df[(df['TIMESTAMP_CST'].dt.hour >= 12) & (df['TIMESTAMP_CST'].dt.hour < 17)]
    
    if df.empty:
        logger.info(f"No data within 12 PM to 5 PM CST for plot {plot_number}")
        conn.close()
        return None
    
    start_time_weather = df['TIMESTAMP'].min()
    end_time_weather = df['TIMESTAMP'].max()
    
    weather_data = get_weather_data(conn, start_time_weather, end_time_weather)
    
    df = df.sort_values('TIMESTAMP')
    weather_data = weather_data.sort_values('TIMESTAMP')
    
    df = pd.merge_asof(df, weather_data, on='TIMESTAMP', direction='nearest')
    
    df['canopy_temp'] = df[irt_column]
    
    logger.info(f"Calculating CWSI for {len(df)} rows")
    df['cwsi_th1'] = df.apply(lambda row: calculate_cwsi_th1(row, CROP_HEIGHT, LAI, LATITUDE, SURFACE_ALBEDO), axis=1)
    df_cwsi = df[['TIMESTAMP', 'cwsi_th1', 'WndAveSpd_3m', 'Solar_2m_Avg']].dropna()
    
    # Ensure TIMESTAMP format is consistent
    df_cwsi['TIMESTAMP'] = df_cwsi['TIMESTAMP'].dt.tz_convert('UTC')
    
    update_cwsi_th1(conn, plot_number, df_cwsi)
    
    # Confirm if values were inserted into DB
    cursor = conn.cursor()
    cursor.execute(f"SELECT COUNT(*) FROM plot_{plot_number} WHERE cwsi_th1 IS NOT NULL")
    count = cursor.fetchone()[0]
    logger.info(f"Number of rows with CWSI values in DB for plot {plot_number}: {count}")
    
    # Perform analytics
    if not df_cwsi.empty:
        max_cwsi = df_cwsi['cwsi_th1'].max()
        min_cwsi = df_cwsi['cwsi_th1'].min()
        avg_cwsi = df_cwsi['cwsi_th1'].mean()
        
        max_cwsi_row = df_cwsi.loc[df_cwsi['cwsi_th1'].idxmax()]
        min_cwsi_row = df_cwsi.loc[df_cwsi['cwsi_th1'].idxmin()]
        
        analytics = {
            'max_cwsi': max_cwsi,
            'min_cwsi': min_cwsi,
            'avg_cwsi': avg_cwsi,
            'max_cwsi_windspeed': max_cwsi_row['WndAveSpd_3m'],
            'max_cwsi_solar_rad': max_cwsi_row['Solar_2m_Avg'],
            'min_cwsi_windspeed': min_cwsi_row['WndAveSpd_3m'],
            'min_cwsi_solar_rad': min_cwsi_row['Solar_2m_Avg'],
        }
        
        logger.info(f"Analytics for plot {plot_number}:")
        for key, value in analytics.items():
            logger.info(f"{key}: {value}")
    
    conn.close()
    
    end_time = time.time()
    duration = end_time - start_time
    logger.info(f"CWSI computation completed for plot {plot_number}. Rows processed: {len(df_cwsi)}")
    logger.info(f"Total execution time: {duration:.2f} seconds")
    
    # Plot the CWSI values
    plot_cwsi(df_cwsi, plot_number)
    
    return analytics

def update_cwsi_th1(conn, plot_number, df_cwsi):
    table_name = f"LINEAR_CORN_trt1_plot_{plot_number}"
    logger.info(f"Updating CWSI-TH1 for {table_name}")

    cursor = conn.cursor()

    # Check if the column exists, if not, create it
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = [column[1] for column in cursor.fetchall()]
    if 'cwsi_th1' not in columns:
        cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN 'cwsi_th1' REAL")
        conn.commit()
        logger.info(f"Added 'cwsi_th1' column to {table_name} table")

    rows_updated = 0
    for _, row in df_cwsi.iterrows():
        timestamp = row['TIMESTAMP'].tz_convert('UTC')
        start_time = timestamp - timedelta(minutes=30)
        end_time = timestamp + timedelta(minutes=30)
        
        query = f"""
        UPDATE {table_name}
        SET 'cwsi_th1' = ?
        WHERE TIMESTAMP BETWEEN ? AND ?
        """
        logger.debug(f"Executing query: {query} with params: {row['cwsi_th1']}, {start_time}, {end_time}")
        
        cursor.execute(query, (row['cwsi_th1'], start_time.strftime('%Y-%m-%d %H:%M:%S'), end_time.strftime('%Y-%m-%d %H:%M:%S')))
        
        if cursor.rowcount > 0:
            rows_updated += cursor.rowcount
            # Immediately fetch and log the updated value
            verify_query = f"""
            SELECT TIMESTAMP, cwsi_th1 FROM {table_name}
            WHERE TIMESTAMP BETWEEN ? AND ?
            """
            cursor.execute(verify_query, (start_time.strftime('%Y-%m-%d %H:%M:%S'), end_time.strftime('%Y-%m-%d %H:%M:%S')))
            updated_data = cursor.fetchone()
            if updated_data:
                logger.info(f"Updated data: Timestamp={updated_data[0]}, CWSI-TH1={updated_data[1]}")
            else:
                logger.warning(f"No data found immediately after update for timestamp: {timestamp}")
        else:
            logger.warning(f"No matching row found for timestamp: {timestamp}")

    conn.commit()
    
    logger.info(f"Successfully updated CWSI-TH1 for {table_name}. Rows updated: {rows_updated}")
    
    # Verify the update with a sample of data
    cursor.execute(f"SELECT TIMESTAMP, cwsi_th1 FROM {table_name} WHERE cwsi_th1 IS NOT NULL LIMIT 5")
    sample_data = cursor.fetchall()
    logger.info(f"Sample of updated data: {sample_data}")

    # Check for any rows that weren't updated
    cursor.execute(f"""
    SELECT COUNT(*) FROM {table_name}
    WHERE 'cwsi_th1' IS NULL
    """)
    unupdated_rows = cursor.fetchone()[0]
    logger.info(f"Rows not updated: {unupdated_rows}")

    return rows_updated

def update_cwsi(conn, plot_number, df_cwsi):
    logger.info(f"Updating CWSI for plot {plot_number}")
    
    cursor = conn.cursor()
    
    # Print columns before changes
    cursor.execute(f"PRAGMA table_info(`LINEAR_CORN_trt1_plot_{plot_number}`)")
    columns_before = [column[1] for column in cursor.fetchall()]
    logger.info(f"Columns before update: {columns_before}")
    
    # Check if the column exists, if not, create it
    if 'cwsi_th1' not in columns_before:
        cursor.execute(f"""
        ALTER TABLE `LINEAR_CORN_trt1_plot_{plot_number}`
        ADD COLUMN cwsi_th1 REAL
        """)
        logger.info(f"Added cwsi_th1 column to `LINEAR_CORN_trt1_plot_{plot_number}`")
    
    # Print columns after potential addition
    cursor.execute(f"PRAGMA table_info(`LINEAR_CORN_trt1_plot_{plot_number}`)")
    columns_after = [column[1] for column in cursor.fetchall()]
    logger.info(f"Columns after potential addition: {columns_after}")
    
    # Reset all cwsi_th1 values to NULL
    cursor.execute(f"""
    UPDATE `LINEAR_CORN_trt1_plot_{plot_number}`
    SET cwsi_th1 = NULL
    """)
    logger.info(f"Reset all cwsi_th1 values to NULL")
    
    # Then, update with new values
    updated_rows = 0
    for _, row in df_cwsi.iterrows():
        cursor.execute(f"""
        UPDATE `LINEAR_CORN_trt1_plot_{plot_number}`
        SET cwsi_th1 = ?
        WHERE TIMESTAMP = ?
        """, (row['cwsi_th1'], row['TIMESTAMP']))
        updated_rows += cursor.rowcount
    
    conn.commit()
    
    logger.info(f"Attempted to update {len(df_cwsi)} rows, actually updated {updated_rows} rows")
    
    # Confirm the update
    cursor.execute(f"SELECT COUNT(*) FROM `LINEAR_CORN_trt1_plot_{plot_number}` WHERE cwsi_th1 IS NOT NULL")
    count = cursor.fetchone()[0]
    logger.info(f"Number of rows with CWSI values in DB for plot {plot_number}: {count}")
    
    # Print sample data
    cursor.execute(f"""
    SELECT TIMESTAMP, cwsi_th1 FROM `LINEAR_CORN_trt1_plot_{plot_number}`
    WHERE cwsi_th1 IS NOT NULL
    LIMIT 5
    """)
    sample_data = cursor.fetchall()
    logger.info(f"Sample data after update: {sample_data}")
    
    # Debug: print a few rows from df_cwsi
    logger.info(f"Sample data from df_cwsi:")
    logger.info(df_cwsi.head().to_string())

def main():
    logger.info(f"Using database at path: {DB_PATH}")
    plot_numbers = ['5006', '5010', '5023']
    all_analytics = {}
    for plot_number in plot_numbers:
        result = compute_cwsi(plot_number)
        print(result)
        
        # Store analytics for each plot
        all_analytics[plot_number] = result
    
    # Compute overall statistics
    all_max_cwsi = max(analytics['max_cwsi'] for analytics in all_analytics.values())
    all_min_cwsi = min(analytics['min_cwsi'] for analytics in all_analytics.values())
    all_avg_cwsi = statistics.mean(analytics['avg_cwsi'] for analytics in all_analytics.values())
    
    logger.info("Overall CWSI Statistics:")
    logger.info(f"Max CWSI across all plots: {all_max_cwsi}")
    logger.info(f"Min CWSI across all plots: {all_min_cwsi}")
    logger.info(f"Average CWSI across all plots: {all_avg_cwsi}")

if __name__ == "__main__":
    main()