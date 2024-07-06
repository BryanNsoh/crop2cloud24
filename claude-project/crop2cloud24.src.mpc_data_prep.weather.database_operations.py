import sqlite3
import pandas as pd
import numpy as np
import logging
from ..config import DB_NAME

logger = logging.getLogger(__name__)

def create_weather_table(conn):
    conn.execute("""
    CREATE TABLE IF NOT EXISTS weather_data (
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
        Ts_bare_10cm_Avg REAL,
        UV_index REAL,
        Visibility REAL,
        Clouds REAL,
        is_forecast INTEGER
    )
    """)
    logger.info("Created weather_data table")

def store_weather_data(weather_df):
    logger.info(f"Storing weather data. Shape: {weather_df.shape}")
    conn = sqlite3.connect(DB_NAME)
    create_weather_table(conn)
    
    weather_df['is_forecast'] = np.where(weather_df['TIMESTAMP'] > pd.Timestamp.now(tz='UTC'), 1, 0)
    weather_df.to_sql('weather_data', conn, if_exists='replace', index=False)
    
    logger.info(f"Stored {len(weather_df)} weather data records")
    
    conn.close()