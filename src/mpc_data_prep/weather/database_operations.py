import sqlite3
import pandas as pd
import numpy as np
import logging
from ..config import DB_NAME

logger = logging.getLogger(__name__)

def create_weather_tables(conn):
    conn.execute("""
    CREATE TABLE IF NOT EXISTS mesonet_data (
        TIMESTAMP TEXT PRIMARY KEY,
        RECORD REAL,
        Ta_2m_Avg REAL,
        TaMax_2m REAL,
        TaMin_2m REAL,
        RH_2m_Avg REAL,
        Dp_2m_Avg REAL,
        WndAveSpd_3m REAL,
        WndAveDir_3m REAL,
        WndMaxSpd5s_3m REAL,
        PresAvg_1pnt5m REAL,
        Rain_1m_Tot REAL,
        Solar_2m_Avg REAL,
        Ts_bare_10cm_Avg REAL
    )
    """)
    
    conn.execute("""
    CREATE TABLE IF NOT EXISTS static_forecast (
        TIMESTAMP TEXT PRIMARY KEY,
        Ta_2m_Avg REAL,
        TaMax_2m REAL,
        TaMin_2m REAL,
        RH_2m_Avg REAL,
        Dp_2m_Avg REAL,
        WndAveSpd_3m REAL,
        WndAveDir_3m REAL,
        WndMaxSpd5s_3m REAL,
        PresAvg_1pnt5m REAL,
        Rain_1m_Tot REAL,
        UV_index REAL,
        Visibility REAL,
        Clouds REAL
    )
    """)
    
    conn.execute("""
    CREATE TABLE IF NOT EXISTS rolling_forecast (
        TIMESTAMP TEXT PRIMARY KEY,
        Ta_2m_Avg REAL,
        TaMax_2m REAL,
        TaMin_2m REAL,
        RH_2m_Avg REAL,
        Dp_2m_Avg REAL,
        WndAveSpd_3m REAL,
        WndAveDir_3m REAL,
        WndMaxSpd5s_3m REAL,
        PresAvg_1pnt5m REAL,
        Rain_1m_Tot REAL,
        UV_index REAL,
        Visibility REAL,
        Clouds REAL
    )
    """)
    
    logger.info("Created mesonet_data, static_forecast, and rolling_forecast tables")

def store_weather_data(mesonet_df, static_forecast_df, rolling_forecast_df):
    logger.info(f"Storing weather data. Shapes - Mesonet: {mesonet_df.shape}, Static Forecast: {static_forecast_df.shape}, Rolling Forecast: {rolling_forecast_df.shape}")
    conn = sqlite3.connect(DB_NAME)
    create_weather_tables(conn)
    
    mesonet_df.to_sql('mesonet_data', conn, if_exists='replace', index=False)
    static_forecast_df.to_sql('static_forecast', conn, if_exists='replace', index=False)
    rolling_forecast_df.to_sql('rolling_forecast', conn, if_exists='replace', index=False)
    
    logger.info(f"Stored {len(mesonet_df)} mesonet records, {len(static_forecast_df)} static forecast records, and {len(rolling_forecast_df)} rolling forecast records")
    
    conn.close()