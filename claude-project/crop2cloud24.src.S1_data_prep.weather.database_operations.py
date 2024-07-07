import sqlite3
import pandas as pd
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
        Ta_2m_Avg_static_forecast REAL,
        TaMax_2m_static_forecast REAL,
        TaMin_2m_static_forecast REAL,
        RH_2m_Avg_static_forecast REAL,
        WndAveSpd_3m_static_forecast REAL,
        WndAveDir_3m_static_forecast REAL,
        WndMaxSpd5s_3m_static_forecast REAL,
        PresAvg_1pnt5m_static_forecast REAL,
        Rain_1m_Tot_static_forecast REAL,
        UV_index_static_forecast REAL,
        Visibility_static_forecast REAL,
        Clouds_static_forecast REAL,
        Ta_2m_Avg_rolling_forecast REAL,
        TaMax_2m_rolling_forecast REAL,
        TaMin_2m_rolling_forecast REAL,
        RH_2m_Avg_rolling_forecast REAL,
        WndAveSpd_3m_rolling_forecast REAL,
        WndAveDir_3m_rolling_forecast REAL,
        WndMaxSpd5s_3m_rolling_forecast REAL,
        PresAvg_1pnt5m_rolling_forecast REAL,
        Rain_1m_Tot_rolling_forecast REAL,
        UV_index_rolling_forecast REAL,
        Visibility_rolling_forecast REAL,
        Clouds_rolling_forecast REAL
    )
    """)
    
    logger.info("Created weather_data table")

def store_weather_data(merged_weather_data: pd.DataFrame):
    logger.info(f"Storing merged weather data. Shape: {merged_weather_data.shape}")
    conn = sqlite3.connect(DB_NAME)
    create_weather_table(conn)
    
    merged_weather_data.to_sql('weather_data', conn, if_exists='replace', index=False)
    
    logger.info(f"Stored {len(merged_weather_data)} weather records")
    
    conn.close()