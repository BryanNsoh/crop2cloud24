import pandas as pd
import numpy as np
from google.cloud import bigquery
from datetime import datetime, timedelta
import pytz
import logging
import sys
import time
import requests
import os
import math
import json

class CustomFormatter(logging.Formatter):
    def format(self, record):
        return f"{datetime.now(pytz.timezone('America/Chicago')).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} CST - {record.levelname} - {record.message}"

logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(CustomFormatter())
logger.addHandler(handler)

STEFAN_BOLTZMANN = 5.67e-8
CP = 1005
GRAVITY = 9.81
K = 0.41
CROP_HEIGHT = 1.6
LATITUDE = 41.15
SURFACE_ALBEDO = 0.23

API_KEY = os.environ.get('NDVI_API_KEY')
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

def get_bigquery_client():
    logger.info("Initializing BigQuery client")
    return bigquery.Client()

def get_irt_tables(client):
    logger.info("Retrieving IRT tables for treatment 1")
    dataset = 'LINEAR_CORN_trt1'
    query = f"""
    SELECT table_name
    FROM `crop2cloud24.{dataset}.INFORMATION_SCHEMA.TABLES`
    WHERE table_name LIKE 'plot_%'
    """
    query_job = client.query(query)
    results = query_job.result()
    
    irt_tables = []
    for row in results:
        table_name = row['table_name']
        schema_query = f"""
        SELECT column_name
        FROM `crop2cloud24.{dataset}.INFORMATION_SCHEMA.COLUMNS`
        WHERE table_name = '{table_name}' AND column_name LIKE 'IRT%' AND column_name NOT LIKE '%_pred'
        """
        schema_job = client.query(schema_query)
        schema_results = schema_job.result()
        if schema_results.total_rows > 0:
            irt_tables.append(f"{dataset}.{table_name}")
    
    logger.info(f"Found {len(irt_tables)} tables with IRT sensors in treatment 1: {irt_tables}")
    return irt_tables

def get_unprocessed_data(client, table_name, irt_column):
    logger.info(f"Retrieving unprocessed data for table {table_name}")
    seven_days_ago = datetime.now(pytz.UTC) - timedelta(days=10)
    query = f"""
    SELECT TIMESTAMP, {irt_column}, is_actual
    FROM `crop2cloud24.{table_name}`
    WHERE TIMESTAMP >= '{seven_days_ago.isoformat()}'
    ORDER BY TIMESTAMP
    """
    df = client.query(query).to_dataframe()
    logger.info(f"Retrieved {len(df)} rows for table {table_name}")
    return df

def get_weather_data(client, start_time, end_time):
    logger.info(f"Retrieving weather data from {start_time} to {end_time}")
    query = f"""
    SELECT *
    FROM `crop2cloud24.weather.current-weather-mesonet`
    WHERE TIMESTAMP BETWEEN '{start_time}' AND '{end_time}'
    ORDER BY TIMESTAMP
    """
    df = client.query(query).to_dataframe()
    logger.info(f"Retrieved {len(df)} weather data rows")
    return df

def update_cwsi(client, table_name, df_cwsi):
    logger.info(f"Updating CWSI for table {table_name}")
    
    seven_days_ago = datetime.now(pytz.UTC) - timedelta(days=7)
    delete_query = f"""
    DELETE FROM `crop2cloud24.{table_name}`
    WHERE TIMESTAMP >= '{seven_days_ago.isoformat()}' AND cwsi IS NOT NULL
    """
    client.query(delete_query).result()
    
    df_cwsi['TIMESTAMP'] = df_cwsi['TIMESTAMP'].apply(lambda x: x.replace(minute=1, second=0, microsecond=0))

    job_config = bigquery.LoadJobConfig(
        schema=[
            bigquery.SchemaField("TIMESTAMP", "TIMESTAMP", mode="REQUIRED"),
            bigquery.SchemaField("cwsi", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("is_actual", "BOOLEAN", mode="REQUIRED"),
        ],
        write_disposition="WRITE_APPEND",
    )
    
    table_ref = client.dataset(table_name.split('.')[0]).table(table_name.split('.')[1])
    
    job = client.load_table_from_dataframe(df_cwsi, table_ref, job_config=job_config)
    job.result()
    
    logger.info(f"Successfully updated CWSI for table {table_name}. Rows processed: {len(df_cwsi)}")

def compute_cwsi(request):
    start_time = time.time()
    logger.info("Starting CWSI computation")
    
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
    
    client = get_bigquery_client()
    irt_tables = get_irt_tables(client)

    total_processed = 0
    for table_name in irt_tables:
        logger.info(f"Processing table: {table_name}")
        
        try:
            schema_query = f"""
            SELECT column_name
            FROM `crop2cloud24.{table_name.split('.')[0]}.INFORMATION_SCHEMA.COLUMNS`
            WHERE table_name = '{table_name.split('.')[1]}' AND column_name LIKE 'IRT%' AND column_name NOT LIKE '%_pred'
            """
            schema_job = client.query(schema_query)
            schema_results = schema_job.result()
            irt_column = next(schema_results)[0]
            logger.info(f"IRT column for table {table_name}: {irt_column}")
            
            df = get_unprocessed_data(client, table_name, irt_column)
            
            if df.empty:
                logger.info(f"No unprocessed data for table {table_name}")
                continue
            
            logger.info(f"Processing {len(df)} rows for table {table_name}")
            
            df['TIMESTAMP_CST'] = df['TIMESTAMP'].dt.tz_convert('America/Chicago')
            df = df[(df['TIMESTAMP_CST'].dt.hour >= 12) & (df['TIMESTAMP_CST'].dt.hour < 17)]
            
            if df.empty:
                logger.info(f"No data within 12 PM to 5 PM CST for table {table_name}")
                continue
            
            start_time_weather = df['TIMESTAMP'].min()
            end_time_weather = df['TIMESTAMP'].max()
            
            weather_data = get_weather_data(client, start_time_weather, end_time_weather)
            
            df = df.sort_values('TIMESTAMP')
            weather_data = weather_data.sort_values('TIMESTAMP')
            
            df = pd.merge_asof(df, weather_data, on='TIMESTAMP', direction='nearest')
            
            df['canopy_temp'] = df[irt_column]
            logger.info(f"Calculating CWSI for {len(df)} rows")
            df['cwsi'] = df.apply(lambda row: calculate_cwsi_th1(row, CROP_HEIGHT, LAI, LATITUDE, SURFACE_ALBEDO), axis=1)
            df_cwsi = df[['TIMESTAMP', 'cwsi', 'is_actual']].dropna()
            
            update_cwsi(client, table_name, df_cwsi)
            
            total_processed += len(df_cwsi)
            logger.info(f"Processed {len(df_cwsi)} rows for table {table_name}")
        
        except Exception as e:
            logger.error(f"Error processing table {table_name}: {str(e)}")
            continue

    end_time = time.time()
    duration = end_time - start_time
    logger.info(f"CWSI computation completed. Total rows processed: {total_processed}")
    logger.info(f"Total execution time: {duration:.2f} seconds")
    return f"CWSI computation completed. Total rows processed: {total_processed}. Execution time: {duration:.2f} seconds"

if __name__ == "__main__":
    compute_cwsi(None)