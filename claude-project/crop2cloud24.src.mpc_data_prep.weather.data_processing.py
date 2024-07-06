import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def log_dataframe_info(df: pd.DataFrame, stage: str):
    logger.info(f"--- DataFrame Info at {stage} ---")
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Columns: {df.columns.tolist()}")
    logger.info(f"Index: {df.index.name}")
    logger.info(f"Data types:\n{df.dtypes}")
    logger.info(f"First few rows:\n{df.head().to_string()}")
    logger.info(f"NaN count:\n{df.isna().sum()}")
    if 'TIMESTAMP' in df.columns:
        logger.info(f"TIMESTAMP column - min: {df['TIMESTAMP'].min()}, max: {df['TIMESTAMP'].max()}")
    elif df.index.name == 'TIMESTAMP':
        logger.info(f"TIMESTAMP index - min: {df.index.min()}, max: {df.index.max()}")
    logger.info("----------------------------")

def create_full_hourly_index(start_time: pd.Timestamp, end_time: pd.Timestamp) -> pd.DatetimeIndex:
    logger.info(f"Creating full hourly index from {start_time} to {end_time}")
    return pd.date_range(start=start_time, end=end_time, freq='h')

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Removing duplicates")
    log_dataframe_info(df, "Before removing duplicates")
    df = df.sort_values('TIMESTAMP').groupby('TIMESTAMP').last().reset_index()
    log_dataframe_info(df, "After removing duplicates")
    return df

def interpolate_hourly(df: pd.DataFrame, full_index: pd.DatetimeIndex) -> pd.DataFrame:
    logger.info("Interpolating hourly data")
    log_dataframe_info(df, "Before interpolation")
    
    df = remove_duplicates(df)
    df = df.set_index('TIMESTAMP')
    df_hourly = df.reindex(full_index)
    
    numeric_columns = df_hourly.select_dtypes(include=['float64', 'int64']).columns
    logger.info(f"Numeric columns for interpolation: {numeric_columns.tolist()}")
    
    df_hourly[numeric_columns] = df_hourly[numeric_columns].interpolate(method='time')
    
    if 'Rain_1m_Tot' in df_hourly.columns:
        df_hourly['Rain_1m_Tot'] = df_hourly['Rain_1m_Tot'].fillna(0)
    
    df_hourly = df_hourly.reset_index()  # Reset index to make TIMESTAMP a column again
    
    log_dataframe_info(df_hourly, "After interpolation")
    return df_hourly

def clip_forecast_data(df: pd.DataFrame, clip_timestamp: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    logger.info(f"Clipping forecast data at {clip_timestamp}")
    log_dataframe_info(df, "Before clipping")
    df_former = df[df['TIMESTAMP'] <= clip_timestamp]
    df_latter = df[df['TIMESTAMP'] > clip_timestamp]
    logger.info(f"Clipped data shapes - Former: {df_former.shape}, Latter: {df_latter.shape}")
    log_dataframe_info(df_former, "Clipped former data")
    log_dataframe_info(df_latter, "Clipped latter data")
    return df_former, df_latter

def combine_weather_data(mesonet_df: pd.DataFrame, static_former: pd.DataFrame, rolling_former: pd.DataFrame) -> pd.DataFrame:
    logger.info("Combining weather data")
    log_dataframe_info(mesonet_df, "Mesonet data before combination")
    log_dataframe_info(static_former, "Static forecast data before combination")
    log_dataframe_info(rolling_former, "Rolling forecast data before combination")
    
    # Ensure 'TIMESTAMP' is a column in all DataFrames
    for df in [mesonet_df, static_former, rolling_former]:
        if 'TIMESTAMP' not in df.columns and df.index.name == 'TIMESTAMP':
            df.reset_index(inplace=True)
    
    combined_df = pd.concat([mesonet_df, static_former, rolling_former], axis=0)
    log_dataframe_info(combined_df, "After concatenation")
    
    combined_df = combined_df.sort_values('TIMESTAMP').drop_duplicates(subset='TIMESTAMP', keep='first')
    log_dataframe_info(combined_df, "After sorting and removing duplicates")
    
    return combined_df

def process_weather_data(weather_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    logger.info("Starting weather data processing")
    
    mesonet_data = weather_data['current-weather-mesonet']
    static_forecast = weather_data['forecast_four_day_static']
    rolling_forecast = weather_data['forecast_four_day_rolling']
    
    log_dataframe_info(mesonet_data, "Mesonet data at start")
    log_dataframe_info(static_forecast, "Static forecast data at start")
    log_dataframe_info(rolling_forecast, "Rolling forecast data at start")

    mesonet_latest_timestamp = mesonet_data['TIMESTAMP'].max()
    logger.info(f"Latest mesonet timestamp: {mesonet_latest_timestamp}")
    
    full_index = create_full_hourly_index(
        min(mesonet_data['TIMESTAMP'].min(), static_forecast['TIMESTAMP'].min(), rolling_forecast['TIMESTAMP'].min()),
        max(static_forecast['TIMESTAMP'].max(), rolling_forecast['TIMESTAMP'].max())
    )
    
    static_former, _ = clip_forecast_data(static_forecast, mesonet_latest_timestamp)
    rolling_former, rolling_latter = clip_forecast_data(rolling_forecast, mesonet_latest_timestamp)

    logger.info("Interpolating hourly data...")
    static_former_hourly = interpolate_hourly(static_former, full_index)
    rolling_former_hourly = interpolate_hourly(rolling_former, full_index)
    mesonet_hourly = interpolate_hourly(mesonet_data, full_index)

    logger.info("Combining data...")
    try:
        combined_data = combine_weather_data(mesonet_hourly, static_former_hourly, rolling_former_hourly)
        log_dataframe_info(combined_data, "Final combined data")
    except Exception as e:
        logger.error(f"Error in combine_weather_data: {str(e)}")
        logger.error("Dumping DataFrames before combination:")
        log_dataframe_info(mesonet_hourly, "Mesonet hourly before combination")
        log_dataframe_info(static_former_hourly, "Static forecast hourly before combination")
        log_dataframe_info(rolling_former_hourly, "Rolling forecast hourly before combination")
        raise

    return combined_data