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
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, HourLocator, DayLocator
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

class CustomFormatter(logging.Formatter):
    def format(self, record):
        record.message = record.getMessage()
        return f"{datetime.now(pytz.timezone('America/Chicago')).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} CST - {record.levelname} - {record.message}"

logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(CustomFormatter())
logger.addHandler(handler)

DB_PATH = 'mpc_data.db'

STEFAN_BOLTZMANN = 5.67e-8
CP = 1005
GRAVITY = 9.81
K = 0.41
CROP_HEIGHT = 1.6
LATITUDE = 41.15
SURFACE_ALBEDO = 0.23

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
    return net_radiation * np.exp(-0.6 * lai)

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
    
    if cwsi < 0 or cwsi > 1.5:
        logger.warning(f"CWSI value out of extended range: {cwsi}")
        return None
    
    return cwsi

def get_db_connection():
    return sqlite3.connect(DB_PATH)

def get_plot_data(conn, plot_number, irt_column):
    query = f"""
    SELECT TIMESTAMP, {irt_column}, is_actual
    FROM plot_{plot_number}
    WHERE TIMESTAMP >= datetime('now', '-10 days')
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
    for _, row in df_cwsi.iterrows():
        cursor.execute(f"""
        UPDATE plot_{plot_number}
        SET cwsi = ?, is_actual = 1
        WHERE TIMESTAMP = ?
        """, (row['cwsi'], row['TIMESTAMP'].strftime('%Y-%m-%d %H:%M:%S')))
    conn.commit()
    
    logger.info(f"Successfully updated CWSI for plot {plot_number}. Rows processed: {len(df_cwsi)}")

def plot_temperatures(df, plot_number):
    # Ensure data is hourly
    df = df.set_index('TIMESTAMP').resample('H').mean().reset_index()
    
    # Create figure and axis objects with subplots()
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Plot data
    ax.plot(df['TIMESTAMP'], df['canopy_temp'], label='Canopy Temperature', color='green')
    ax.plot(df['TIMESTAMP'], df['Ta_2m_Avg'], label='Air Temperature', color='red')
    
    # Set title and labels
    ax.set_title(f'Hourly Temperatures for Plot {plot_number}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Temperature (°C)')
    
    # Set x-axis major ticks to midnight of each day
    ax.xaxis.set_major_locator(DayLocator())
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    
    # Set x-axis minor ticks to show hours
    ax.xaxis.set_minor_locator(HourLocator(interval=6))
    ax.xaxis.set_minor_formatter(DateFormatter('%H:00'))
    
    # Rotate and align the tick labels so they look better
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    plt.setp(ax.xaxis.get_minorticklabels(), rotation=45, ha='right')
    
    # Use a more fine-grained grid
    ax.grid(which='both', linestyle=':', linewidth='0.5', color='gray')
    
    # Add legend
    ax.legend()
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join('images', f'temperatures_plot_{plot_number}.png'))
    plt.close()

def plot_precipitation(df, plot_number):
    # Ensure data is hourly and calculate cumulative precipitation
    df = df.set_index('TIMESTAMP').resample('H').agg({'Rain_Tot': 'sum', 'TIMESTAMP': 'first'}).reset_index(drop=True)
    df['Cumulative_Rain'] = df['Rain_Tot'].cumsum()
    
    fig, ax1 = plt.subplots(figsize=(15, 8))
    
    # Plot hourly precipitation
    ax1.bar(df['TIMESTAMP'], df['Rain_Tot'], width=1/24, align='center', label='Hourly Precipitation', color='blue', alpha=0.6)
    ax1.set_ylabel('Hourly Precipitation (mm)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # Plot cumulative precipitation on secondary y-axis
    ax2 = ax1.twinx()
    ax2.plot(df['TIMESTAMP'], df['Cumulative_Rain'], color='red', label='Cumulative Precipitation')
    ax2.set_ylabel('Cumulative Precipitation (mm)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    plt.title(f'Precipitation for Plot {plot_number}')
    ax1.set_xlabel('Date')
    
    # Set x-axis major ticks to midnight of each day
    ax1.xaxis.set_major_locator(DayLocator())
    ax1.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    
    # Set x-axis minor ticks to show hours
    ax1.xaxis.set_minor_locator(HourLocator(interval=6))
    ax1.xaxis.set_minor_formatter(DateFormatter('%H:00'))
    
    # Rotate and align the tick labels so they look better
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    plt.setp(ax1.xaxis.get_minorticklabels(), rotation=45, ha='right')
    
    # Use a more fine-grained grid
    ax1.grid(which='both', linestyle=':', linewidth='0.5', color='gray')
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join('images', f'precipitation_plot_{plot_number}.png'))
    plt.close()

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
    
    # Create 'images' directory if it doesn't exist
    os.makedirs('images', exist_ok=True)
    
    # Plot temperatures and precipitation before CWSI computation
    plot_temperatures(df, plot_number)
    plot_precipitation(df, plot_number)
    
    logger.info(f"Calculating CWSI for {len(df)} rows")
    df['cwsi'] = df.apply(lambda row: calculate_cwsi_th1(row, CROP_HEIGHT, LAI, LATITUDE, SURFACE_ALBEDO), axis=1)
    df_cwsi = df[['TIMESTAMP', 'cwsi', 'is_actual']].dropna()
    
    update_cwsi(conn, plot_number, df_cwsi)
    
    conn.close()
    
    # Plot CWSI results
    plt.figure(figsize=(15, 8))
    plt.scatter(df_cwsi['TIMESTAMP'], df_cwsi['cwsi'], label='CWSI', alpha=0.6)
    plt.title(f'CWSI for Plot {plot_number}')
    plt.xlabel('Date')
    plt.ylabel('CWSI')
    plt.legend()
    plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(DayLocator())
    plt.gca().xaxis.set_minor_locator(HourLocator(interval=6))
    plt.gca().xaxis.set_minor_formatter(DateFormatter('%H:00'))
    plt.setp(plt.gca().xaxis.get_majorticklabels(), rotation=45, ha='right')
    plt.setp(plt.gca().xaxis.get_minorticklabels(), rotation=45, ha='right')
    plt.ylim(0, 1.5)  # Set y-axis limits to 0-1.5 for CWSI
    plt.grid(which='both', linestyle=':', linewidth='0.5', color='gray')
    plt.tight_layout()
    plt.savefig(os.path.join('images', f'cwsi_plot_{plot_number}.png'))
    plt.close()
    
    end_time = time.time()
    duration = end_time - start_time
    logger.info(f"CWSI computation completed for plot {plot_number}. Rows processed: {len(df_cwsi)}")
    logger.info(f"Total execution time: {duration:.2f} seconds")
    return f"CWSI computation completed for plot {plot_number}. Rows processed: {len(df_cwsi)}. Execution time: {duration:.2f} seconds"

def main():
    plot_numbers = ['5006', '5010', '5023']
    for plot_number in plot_numbers:
        result = compute_cwsi(plot_number)
        print(result)

if __name__ == "__main__":
    main()