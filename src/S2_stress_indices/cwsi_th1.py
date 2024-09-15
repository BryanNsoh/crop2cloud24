import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import logging
import requests
import math
import json
from dotenv import load_dotenv
import statistics 
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

def get_all_irt_tables(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name != 'weather_data'")
    tables = [table[0] for table in cursor.fetchall()]
    
    logger.info("Selected IRT tables:")
    for table in tables:
        logger.info(f"  {table}")
    
    return tables

def get_irt_columns(conn, table_name):
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = [column[1] for column in cursor.fetchall()]
    irt_columns = [col for col in columns if 'irt' in col.lower()]
    
    logger.info(f"Found IRT columns for {table_name}:")
    for col in irt_columns:
        logger.info(f"  {col}")
    
    return irt_columns

def compute_cwsi(conn, table_name):
    logger.info(f"Starting CWSI computation for {table_name}")
    
    polygon_id = get_or_create_polygon()
    if polygon_id is None:
        logger.error("Failed to get or create polygon. Aborting CWSI computation.")
        return None
    
    latest_ndvi = get_latest_ndvi(polygon_id)
    if latest_ndvi is None:
        logger.error("Failed to retrieve NDVI data. Aborting CWSI computation.")
        return None
    
    LAI = calculate_lai(latest_ndvi)
    logger.info(f"Using NDVI: {latest_ndvi}, Calculated LAI: {LAI}")
    
    irt_columns = get_irt_columns(conn, table_name)
    if not irt_columns:
        logger.warning(f"No IRT columns found for {table_name}")
        return None
    
    # Modify to fetch only last 5 days of data
    end_time = datetime.now()
    start_time = end_time - timedelta(days=5)
    
    logger.info(f"Querying data for {table_name} from {start_time} to {end_time}")
    query = f"""
    SELECT TIMESTAMP, {', '.join(irt_columns)}, Ta_2m_Avg, RH_2m_Avg, Solar_2m_Avg, WndAveSpd_3m, PresAvg_1pnt5m
    FROM {table_name}
    WHERE TIMESTAMP BETWEEN ? AND ?
    ORDER BY TIMESTAMP
    """
    df = pd.read_sql_query(query, conn, params=(start_time, end_time), parse_dates=['TIMESTAMP'])

    if df.empty:
        logger.info(f"No data found for {table_name} in the specified date range")
        return None
    
    logger.info(f"Retrieved {len(df)} rows of data for {table_name}")
    
    # Ensure we're working with hourly data
    df = df.set_index('TIMESTAMP').resample('h').mean().reset_index()
    
    # Deduplicate the data focusing on non null columns
    df = df.drop_duplicates(subset=[col for col in df.columns if col != 'TIMESTAMP'])
    logger.info(f"Data deduplicated. {len(df)} rows remaining after deduplication.")
    
    
    
    # Convert UTC to CST for filtering
    df['TIMESTAMP_CST'] = df['TIMESTAMP'].dt.tz_convert('America/Chicago')
    df = df[(df['TIMESTAMP_CST'].dt.hour >= 12) & (df['TIMESTAMP_CST'].dt.hour < 17)]
    
    
    
    if df.empty:
        logger.info(f"No data within 12 PM to 5 PM CST for {table_name}")
        return None
    
    for irt_column in irt_columns:
        logger.info(f"Processing IRT column: {irt_column}")
        df['canopy_temp'] = df[irt_column]
        
        #print how many non nan values are in canopy temp and in the irt column
        print(f"Number of non nan values in canopy temp: {df['canopy_temp'].notna().sum()}")
        print(f"Number of non nan values in {irt_column}: {df[irt_column].notna().sum()}")
        
        # Filter out rows where any of the following in addition to irt is null (we want rows where values for all are present. be tolerant to variations of timestamp within the hour): ['canopy_temp', 'Ta_2m_Avg', 'RH_2m_Avg', 'Solar_2m_Avg', 'WndAveSpd_3m', 'PresAvg_1pnt5m']. Print stats to show how the process goes and how much data is left
        required_columns = ['canopy_temp', 'Ta_2m_Avg', 'RH_2m_Avg', 'Solar_2m_Avg', 'WndAveSpd_3m', 'PresAvg_1pnt5m']
        # keep only required columns
        df_valid = df[required_columns + [irt_column]]
        # drop rows with null values in required columns
        
        
        
        df_valid = df_valid.dropna(subset=['canopy_temp'])
        
        print(df_valid.head())  
        
        logger.info(f"Filtered data: {len(df_valid)} rows remaining out of {len(df)} after dropping rows with null values in required columns")
        
        #PRINT df stats
        print(df_valid.describe())
        
        if df_valid.empty:
            logger.warning(f"No valid data for CWSI calculation in {irt_column}")
            continue
        
        logger.info(f"Calculating CWSI for {len(df_valid)} rows")
        df_valid[f'cwsi_th1_{irt_column}'] = df_valid.apply(lambda row: calculate_cwsi_th1(row, CROP_HEIGHT, LAI, LATITUDE, SURFACE_ALBEDO), axis=1)
        
        df_cwsi = df_valid[['TIMESTAMP', f'cwsi_th1', 'WndAveSpd_3m', 'Solar_2m_Avg']].dropna()
        
        # Ensure TIMESTAMP format is consistent
        df_cwsi['TIMESTAMP'] = df_cwsi['TIMESTAMP'].dt.tz_convert('UTC')
        
        update_cwsi_th1(conn, table_name, df_cwsi, irt_column)
        
        # Perform analytics and plotting for each IRT column
        if not df_cwsi.empty:
            perform_analytics(df_cwsi, table_name, irt_column)
            plot_cwsi(df_cwsi, table_name, irt_column)
    
    logger.info(f"CWSI computation completed for {table_name}")
    
    try:
        return df_cwsi
    except UnboundLocalError:
        logger.error("df_cwsi is not defined. Moving on to the next IRT column.")
        return None

def update_cwsi_th1(conn, table_name, df_cwsi, irt_column):
    logger.info(f"Updating CWSI-TH1 for {table_name} and {irt_column}")

    cursor = conn.cursor()
    column_name = f'cwsi_th1_{irt_column}'

    # Check if the column exists, if not, create it
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = [column[1] for column in cursor.fetchall()]
    if column_name not in columns:
        cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN '{column_name}' REAL")
        conn.commit()
        logger.info(f"Added '{column_name}' column to {table_name} table")

    rows_updated = 0
    for _, row in df_cwsi.iterrows():
        cursor.execute(f"""
            UPDATE {table_name}
            SET {column_name} = ?
            WHERE TIMESTAMP = ?
        """, (row[column_name], row['TIMESTAMP']))
        rows_updated += cursor.rowcount

    conn.commit()
    logger.info(f"Updated {rows_updated} rows in {table_name} for {irt_column}")

def perform_analytics(df_cwsi, table_name, irt_column):
    cwsi_column = f'cwsi_th1_{irt_column}'
    max_cwsi = df_cwsi[cwsi_column].max()
    min_cwsi = df_cwsi[cwsi_column].min()
    avg_cwsi = df_cwsi[cwsi_column].mean()
    
    max_cwsi_row = df_cwsi.loc[df_cwsi[cwsi_column].idxmax()]
    min_cwsi_row = df_cwsi.loc[df_cwsi[cwsi_column].idxmin()]
    
    analytics = {
        'max_cwsi': max_cwsi,
        'min_cwsi': min_cwsi,
        'avg_cwsi': avg_cwsi,
        'max_cwsi_windspeed': max_cwsi_row['WndAveSpd_3m'],
        'max_cwsi_solar_rad': max_cwsi_row['Solar_2m_Avg'],
        'min_cwsi_windspeed': min_cwsi_row['WndAveSpd_3m'],
        'min_cwsi_solar_rad': min_cwsi_row['Solar_2m_Avg'],
    }
    
    logger.info(f"Analytics for {table_name} - {irt_column}:")
    for key, value in analytics.items():
        logger.info(f"{key}: {value}")

def plot_cwsi(df_cwsi, table_name, irt_column):
    cwsi_column = f'cwsi_th1_{irt_column}'
    plt.figure(figsize=(12, 6))
    plt.plot(df_cwsi['TIMESTAMP'], df_cwsi[cwsi_column], marker='o')
    plt.title(f'CWSI-TH1 for {table_name} - {irt_column}')
    plt.xlabel('Timestamp')
    plt.ylabel('CWSI-TH1')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plots_dir = 'cwsi_plots'
    os.makedirs(plots_dir, exist_ok=True)
    
    plot_filename = f'{plots_dir}/cwsi_plot_{table_name}_{irt_column}.png'
    plt.savefig(plot_filename)
    plt.close()
    
    logger.info(f"CWSI plot saved as {plot_filename}")

def main():
    conn = get_db_connection()
    try:
        irt_tables = get_all_irt_tables(conn)
        logger.info(f"Found {len(irt_tables)} tables in total")
        
        for table_name in irt_tables:
            logger.info(f"Processing {table_name}")
            compute_cwsi(conn, table_name)
    finally:
        conn.close()

if __name__ == "__main__":
    main()