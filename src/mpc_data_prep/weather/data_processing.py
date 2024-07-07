import pandas as pd
import numpy as np
from google.cloud import bigquery
import logging
from datetime import datetime, timedelta
import pytz

logger = logging.getLogger(__name__)

def resample_mesonet_data(mesonet_data: pd.DataFrame) -> pd.DataFrame:
    """
    Resample mesonet data to hourly intervals.
    Average all columns except 'Rain_1m_Tot', which is summed.
    """
    logger.info("Resampling mesonet data to hourly intervals")
    
    # Set TIMESTAMP as index for resampling
    mesonet_data = mesonet_data.set_index('TIMESTAMP')
    
    # Identify columns to average (all except 'Rain_1m_Tot')
    columns_to_average = [col for col in mesonet_data.columns if col != 'Rain_1m_Tot']
    
    # Create a dictionary for resampling operations
    resampling_dict = {col: 'mean' for col in columns_to_average}
    resampling_dict['Rain_1m_Tot'] = 'sum'
    
    # Resample data
    resampled_data = mesonet_data.resample('H').agg(resampling_dict)
    
    # Reset index to make TIMESTAMP a column again
    resampled_data = resampled_data.reset_index()
    
    logger.info(f"Original mesonet data shape: {mesonet_data.shape}")
    logger.info(f"Resampled mesonet data shape: {resampled_data.shape}")
    
    return resampled_data

def align_forecast_timestamps(forecast_df: pd.DataFrame, latest_mesonet_timestamp: pd.Timestamp) -> pd.DataFrame:
    """
    Align forecast timestamps with mesonet data, decrementing until the latest forecast timestamp
    is flush with the latest mesonet timestamp.
    """
    logger.info(f"Original forecast timestamp range: {forecast_df['TIMESTAMP'].min()} to {forecast_df['TIMESTAMP'].max()}")
    
    time_difference = forecast_df['TIMESTAMP'].max() - latest_mesonet_timestamp
    hours_to_subtract = time_difference.total_seconds() / 3600
    logger.info(f"Hours to subtract from forecast timestamps: {hours_to_subtract}")
    
    forecast_df['TIMESTAMP'] = forecast_df['TIMESTAMP'] - pd.Timedelta(hours=hours_to_subtract)
    
    logger.info(f"Adjusted forecast timestamp range: {forecast_df['TIMESTAMP'].min()} to {forecast_df['TIMESTAMP'].max()}")
    
    # Log a sample of adjusted timestamps
    sample_timestamps = forecast_df['TIMESTAMP'].sample(min(5, len(forecast_df))).sort_values()
    logger.info(f"Sample of adjusted forecast timestamps:\n{sample_timestamps.to_string()}")
    
    return forecast_df

def merge_weather_data(mesonet_data: pd.DataFrame, static_forecast: pd.DataFrame, rolling_forecast: pd.DataFrame) -> pd.DataFrame:
    logger.info("Merging weather data")
    
    # Ensure all timestamps are in UTC, convert to timezone-naive, and have nanosecond precision
    for df in [mesonet_data, static_forecast, rolling_forecast]:
        df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], utc=True).dt.tz_convert('UTC').dt.tz_localize(None).astype('datetime64[ns]')
    
    # Get the latest mesonet timestamp
    latest_mesonet_timestamp = mesonet_data['TIMESTAMP'].max()
    logger.info(f"Latest mesonet timestamp: {latest_mesonet_timestamp}")
    logger.info(f"Mesonet data range: {mesonet_data['TIMESTAMP'].min()} to {mesonet_data['TIMESTAMP'].max()}")
    
    # Log number of matching timestamps before alignment
    matches_before = sum(mesonet_data['TIMESTAMP'].dt.floor('h').isin(static_forecast['TIMESTAMP'].dt.floor('h')))
    logger.info(f"Number of matching timestamps (same hour) before alignment: {matches_before}")
    
    # Align forecast timestamps
    static_forecast = align_forecast_timestamps(static_forecast, latest_mesonet_timestamp)
    rolling_forecast = align_forecast_timestamps(rolling_forecast, latest_mesonet_timestamp)
    
    # Log number of matching timestamps after alignment
    matches_after = sum(mesonet_data['TIMESTAMP'].dt.floor('h').isin(static_forecast['TIMESTAMP'].dt.floor('h')))
    logger.info(f"Number of matching timestamps (same hour) after alignment: {matches_after}")
    
    # Print common datapoints within the same hour range
    common_hours = set(mesonet_data['TIMESTAMP'].dt.floor('h')) & set(static_forecast['TIMESTAMP'].dt.floor('h'))
    logger.info(f"Common hours between mesonet and forecast data: {sorted(common_hours)}")
    
    # Add suffixes to forecast columns
    static_forecast_columns = {col: f"{col}_static_forecast" for col in static_forecast.columns if col != "TIMESTAMP"}
    rolling_forecast_columns = {col: f"{col}_rolling_forecast" for col in rolling_forecast.columns if col != "TIMESTAMP"}
    
    static_forecast = static_forecast.rename(columns=static_forecast_columns)
    rolling_forecast = rolling_forecast.rename(columns=rolling_forecast_columns)
    
    # Merge dataframes
    merged_data = pd.merge_asof(
        mesonet_data, 
        static_forecast, 
        on='TIMESTAMP',
        direction='nearest',
        tolerance=pd.Timedelta('1h')
    )
    merged_data = pd.merge_asof(
        merged_data, 
        rolling_forecast, 
        on='TIMESTAMP',
        direction='nearest',
        tolerance=pd.Timedelta('1h')
    )
    
    logger.info(f"Merged data range: {merged_data['TIMESTAMP'].min()} to {merged_data['TIMESTAMP'].max()}")
    logger.info(f"Total rows in merged data: {len(merged_data)}")
    
    # Log number of non-null datapoints in merged table
    non_null_counts = merged_data.notna().sum()
    logger.info(f"Number of non-null datapoints in merged table:")
    for col, count in non_null_counts.items():
        logger.info(f"{col}: {count}")
    
    return merged_data

def process_weather_data(weather_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    logger.info("Starting weather data processing")
    
    mesonet_data = weather_data['current-weather-mesonet']
    static_forecast = weather_data['forecast_four_day_static']
    rolling_forecast = weather_data['forecast_four_day_rolling']
    
    # Remove duplicates
    mesonet_data = mesonet_data.sort_values('TIMESTAMP').drop_duplicates(subset='TIMESTAMP', keep='last')
    static_forecast = static_forecast.sort_values('TIMESTAMP').drop_duplicates(subset='TIMESTAMP', keep='last')
    rolling_forecast = rolling_forecast.sort_values('TIMESTAMP').drop_duplicates(subset='TIMESTAMP', keep='last')
    
    logger.info(f"Mesonet data shape after removing duplicates: {mesonet_data.shape}")
    logger.info(f"Static forecast data shape after removing duplicates: {static_forecast.shape}")
    logger.info(f"Rolling forecast data shape after removing duplicates: {rolling_forecast.shape}")

    # Resample mesonet data to hourly intervals
    mesonet_data = resample_mesonet_data(mesonet_data)

    # Merge weather data
    merged_data = merge_weather_data(mesonet_data, static_forecast, rolling_forecast)

    return merged_data