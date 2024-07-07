import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import pytz
from google.cloud import bigquery

logger = logging.getLogger(__name__)

PROJECT_ID = "crop2cloud24"
HISTORICAL_DAYS = 30
client = bigquery.Client()

def handle_timestamp(df: pd.DataFrame, action: str = 'ensure') -> pd.DataFrame:
    """
    Unified function to handle timestamp-related transformations.
    
    Args:
    df (pd.DataFrame): Input DataFrame
    action (str): Action to perform. Options: 'ensure', 'to_index', 'from_index'
    
    Returns:
    pd.DataFrame: DataFrame with timestamp handled as specified
    """
    if action == 'ensure':
        if 'TIMESTAMP' not in df.columns and df.index.name != 'TIMESTAMP':
            if 'index' in df.columns and pd.api.types.is_datetime64_any_dtype(df['index']):
                df = df.rename(columns={'index': 'TIMESTAMP'})
            else:
                logger.error("DataFrame does not have a TIMESTAMP column or index")
                raise ValueError("DataFrame must have a TIMESTAMP column or index")
        elif df.index.name == 'TIMESTAMP':
            df = df.reset_index()
    elif action == 'to_index':
        df = handle_timestamp(df, 'ensure')
        df = df.set_index('TIMESTAMP')
    elif action == 'from_index':
        if df.index.name == 'TIMESTAMP':
            df = df.reset_index()
    else:
        logger.error(f"Invalid action '{action}' in handle_timestamp")
        raise ValueError(f"Invalid action '{action}' in handle_timestamp")
    
    return df

def log_dataframe_info(df: pd.DataFrame, stage: str):
    logger.info(f"--- DataFrame Info at {stage} ---")
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Columns: {df.columns.tolist()}")
    logger.info(f"Index: {df.index.name}")
    logger.info(f"Data types:\n{df.dtypes}")
    logger.info(f"First few rows:\n{df.head().to_string()}")
    logger.info(f"NaN count:\n{df.isna().sum()}")
    df = handle_timestamp(df, 'ensure')
    logger.info(f"TIMESTAMP column - min: {df['TIMESTAMP'].min()}, max: {df['TIMESTAMP'].max()}")
    logger.info("----------------------------")

def create_full_hourly_index(start_time: pd.Timestamp, end_time: pd.Timestamp) -> pd.DatetimeIndex:
    logger.info(f"Creating full hourly index from {start_time} to {end_time}")
    return pd.date_range(start=start_time, end=end_time, freq='h', tz='UTC')

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    logger.info(f"Removing duplicates. Initial shape: {df.shape}")
    df = handle_timestamp(df, 'ensure')
    df = df.sort_values('TIMESTAMP').groupby('TIMESTAMP', as_index=False).last()
    logger.info(f"After removing duplicates. Shape: {df.shape}")
    return df

def interpolate_hourly(df: pd.DataFrame, full_index: pd.DatetimeIndex) -> pd.DataFrame:
    logger.info(f"Interpolating hourly data for DataFrame with shape {df.shape}")
    
    df = handle_timestamp(df, 'ensure')
    df = remove_duplicates(df)
    
    logger.info(f"Columns before setting index: {df.columns.tolist()}")
    df = df.set_index('TIMESTAMP')
    logger.info(f"Index name after setting: {df.index.name}")
    
    df_hourly = df.reindex(full_index)
    
    numeric_columns = df_hourly.select_dtypes(include=['float64', 'int64']).columns
    logger.info(f"Numeric columns for interpolation: {numeric_columns.tolist()}")
    
    df_hourly[numeric_columns] = df_hourly[numeric_columns].interpolate(method='time')
    
    if 'Rain_1m_Tot' in df_hourly.columns:
        df_hourly['Rain_1m_Tot'] = df_hourly['Rain_1m_Tot'].fillna(0)
    
    df_hourly = df_hourly.reset_index()
    df_hourly = df_hourly.rename(columns={'index': 'TIMESTAMP'})  # Ensure the column is named 'TIMESTAMP'
    
    log_dataframe_info(df_hourly, "After interpolation")
    return df_hourly

def clip_forecast_data(df: pd.DataFrame, clip_timestamp: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    logger.info(f"Clipping forecast data at {clip_timestamp}")
    log_dataframe_info(df, "Before clipping")
    df = handle_timestamp(df, 'ensure')
    df_former = df[df['TIMESTAMP'] <= clip_timestamp]
    df_latter = df[df['TIMESTAMP'] > clip_timestamp]
    logger.info(f"Clipped data shapes - Former: {df_former.shape}, Latter: {df_latter.shape}")
    log_dataframe_info(df_former, "Clipped former data")
    log_dataframe_info(df_latter, "Clipped latter data")
    return df_former, df_latter

def combine_weather_data(mesonet_df: pd.DataFrame, static_forecast: pd.DataFrame, rolling_forecast: pd.DataFrame, mesonet_latest_timestamp: pd.Timestamp) -> pd.DataFrame:
    logger.info("Starting weather data combination process")
    logger.info(f"Mesonet latest timestamp: {mesonet_latest_timestamp}")
    
    log_dataframe_info(mesonet_df, "Mesonet data")
    log_dataframe_info(static_forecast, "Static forecast data")
    log_dataframe_info(rolling_forecast, "Rolling forecast data")
    
    # Clip forecast data
    static_former, static_latter = clip_forecast_data(static_forecast, mesonet_latest_timestamp)
    rolling_former, rolling_latter = clip_forecast_data(rolling_forecast, mesonet_latest_timestamp)
    
    logger.info("Discarding static latter data as per instructions")
    
    # Add suffixes to distinguish between static and rolling forecasts
    static_columns = [col for col in static_former.columns if col != 'TIMESTAMP']
    rolling_columns = [col for col in rolling_former.columns if col != 'TIMESTAMP']
    
    static_former = static_former.rename(columns={col: f"{col}_static_forecast" for col in static_columns})
    rolling_former = rolling_former.rename(columns={col: f"{col}_rolling_forecast" for col in rolling_columns})
    rolling_latter = rolling_latter.rename(columns={col: f"{col}_rolling_forecast" for col in rolling_columns})
    
    # Merge mesonet data with former forecast data
    combined_df = pd.merge(mesonet_df, static_former, on='TIMESTAMP', how='outer')
    combined_df = pd.merge(combined_df, rolling_former, on='TIMESTAMP', how='outer')
    log_dataframe_info(combined_df, "After merging mesonet and former forecast data")
    
    # Process rolling latter data
    rolling_latter['TIMESTAMP'] = rolling_latter['TIMESTAMP'] - timedelta(hours=1)
    rolling_latter = rolling_latter[rolling_latter['TIMESTAMP'] > mesonet_latest_timestamp]
    log_dataframe_info(rolling_latter, "Processed rolling latter data")
    
    # Append rolling latter data
    combined_df = pd.concat([combined_df, rolling_latter]).sort_values('TIMESTAMP')
    log_dataframe_info(combined_df, "Final combined data")
    
    # Verify the combination process
    actual_columns = [col for col in combined_df.columns if not col.endswith('_forecast') and col != 'TIMESTAMP']
    static_forecast_columns = [col for col in combined_df.columns if col.endswith('_static_forecast')]
    rolling_forecast_columns = [col for col in combined_df.columns if col.endswith('_rolling_forecast')]
    
    logger.info(f"Number of actual data columns: {len(actual_columns)}")
    logger.info(f"Number of static forecast columns: {len(static_forecast_columns)}")
    logger.info(f"Number of rolling forecast columns: {len(rolling_forecast_columns)}")
    logger.info(f"Total number of rows: {len(combined_df)}")
    logger.info(f"Number of rows with actual data: {combined_df[actual_columns].notna().any(axis=1).sum()}")
    logger.info(f"Number of rows with static forecast data: {combined_df[static_forecast_columns].notna().any(axis=1).sum()}")
    logger.info(f"Number of rows with rolling forecast data: {combined_df[rolling_forecast_columns].notna().any(axis=1).sum()}")
    
    if len(static_forecast_columns) > 0 and len(rolling_forecast_columns) > 0:
        logger.info("SUCCESS: Both static and rolling forecast columns have been successfully added to the weather table.")
    elif len(static_forecast_columns) > 0:
        logger.warning("PARTIAL SUCCESS: Only static forecast columns were added to the weather table.")
    elif len(rolling_forecast_columns) > 0:
        logger.warning("PARTIAL SUCCESS: Only rolling forecast columns were added to the weather table.")
    else:
        logger.error("FAILURE: No forecast columns were added to the weather table.")
    
    return combined_df

def get_data_with_history(table_name):
    end_time = datetime.now(pytz.UTC)
    start_time = end_time - timedelta(days=HISTORICAL_DAYS)
    
    query = f"""
    SELECT *
    FROM `{PROJECT_ID}.weather.{table_name}`
    WHERE TIMESTAMP BETWEEN '{start_time}' AND '{end_time}'
    ORDER BY TIMESTAMP
    """
    
    logger.info(f"Executing query for {table_name}:\n{query}")
    df = client.query(query).to_dataframe()
    return df

def process_weather_data(weather_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    logger.info("Starting weather data processing")
    
    mesonet_data = weather_data['current-weather-mesonet']
    static_forecast = weather_data['forecast_four_day_static']
    rolling_forecast = weather_data['forecast_four_day_rolling']
    
    log_dataframe_info(mesonet_data, "Mesonet data at start")
    log_dataframe_info(static_forecast, "Static forecast data at start")
    log_dataframe_info(rolling_forecast, "Rolling forecast data at start")

    mesonet_data = handle_timestamp(mesonet_data, 'ensure')
    mesonet_latest_timestamp = mesonet_data['TIMESTAMP'].max()
    logger.info(f"Latest mesonet timestamp: {mesonet_latest_timestamp}")
    
    full_index = create_full_hourly_index(
        min(mesonet_data['TIMESTAMP'].min(), static_forecast['TIMESTAMP'].min(), rolling_forecast['TIMESTAMP'].min()),
        max(static_forecast['TIMESTAMP'].max(), rolling_forecast['TIMESTAMP'].max())
    )
    
    logger.info("Interpolating hourly data...")
    mesonet_hourly = interpolate_hourly(mesonet_data, full_index)
    static_forecast_hourly = interpolate_hourly(static_forecast, full_index)
    rolling_forecast_hourly = interpolate_hourly(rolling_forecast, full_index)

    logger.info("Combining data...")
    try:
        combined_data = combine_weather_data(mesonet_hourly, static_forecast_hourly, rolling_forecast_hourly, mesonet_latest_timestamp)
        log_dataframe_info(combined_data, "Final combined data")
    except Exception as e:
        logger.error(f"Error in combine_weather_data: {str(e)}")
        logger.error("Dumping DataFrames before combination:")
        log_dataframe_info(mesonet_hourly, "Mesonet hourly before combination")
        log_dataframe_info(static_forecast_hourly, "Static forecast hourly before combination")
        log_dataframe_info(rolling_forecast_hourly, "Rolling forecast hourly before combination")
        raise

    return combined_data
