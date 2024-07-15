# src/S1_data_prep/sensor/data_processing.py

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def process_plot_data(plot_data, weather_data):
    processed_data = {}
    
    # Convert weather data TIMESTAMP to nanosecond precision once
    weather_data['TIMESTAMP'] = pd.to_datetime(weather_data['TIMESTAMP'], utc=True).astype('datetime64[ns, UTC]')
    
    for plot_number, df in plot_data.items():
        logger.info(f"Processing data for plot {plot_number}")
        
        # Determine actual data range
        actual_start = df['TIMESTAMP'].min()
        actual_end = df['TIMESTAMP'].max()
        logger.info(f"Actual data range for plot {plot_number}: {actual_start} to {actual_end}")
        
        # Remove duplicate timestamps
        df = df.sort_values('TIMESTAMP').drop_duplicates(subset='TIMESTAMP', keep='first')
        
        # Convert plot data TIMESTAMP to nanosecond precision
        df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], utc=True).astype('datetime64[ns, UTC]')
        
        # Resample to hourly intervals without interpolation
        df = resample_hourly(df)
        
        # Log timestamp dtypes for debugging
        logger.info(f"Plot data TIMESTAMP dtype: {df['TIMESTAMP'].dtype}")
        logger.info(f"Weather data TIMESTAMP dtype: {weather_data['TIMESTAMP'].dtype}")
        
        # Filter weather data to the actual data range
        weather_data_filtered = weather_data[(weather_data['TIMESTAMP'] >= actual_start) & (weather_data['TIMESTAMP'] <= actual_end)]
        
        # Perform the merge
        df = pd.merge_asof(df, weather_data_filtered, on='TIMESTAMP', direction='nearest', tolerance=pd.Timedelta('1h'))
        
        # Drop all-null columns except 'swsi', 'et', and 'cwsi-th2'
        cols_to_keep = ['TIMESTAMP', 'swsi', 'et', 'cwsi-th2']
        null_columns = df.columns[df.isnull().all()].tolist()
        cols_to_drop = [col for col in null_columns if col not in cols_to_keep]
        df = df.drop(columns=cols_to_drop)
        
        # Explicitly drop 'is_actual' column if it exists
        if 'is_actual' in df.columns:
            df = df.drop(columns=['is_actual'])
        
        # Rename 'cwsi-th1' to 'cwsi-th2' if it exists, otherwise add it as a null column
        if 'cwsi-th1' in df.columns:
            df = df.rename(columns={'cwsi-th1': 'cwsi-th2'})
        else:
            df['cwsi-th2'] = np.nan
        
        # Remove other CWSI columns
        cwsi_columns_to_remove = [col for col in df.columns if col.startswith('cwsi') and col != 'cwsi-th2']
        df = df.drop(columns=cwsi_columns_to_remove)
        
        processed_data[plot_number] = df
        
        logger.info(f"Processed data for plot {plot_number}. Shape: {df.shape}")
        logger.info(f"Processed data range: {df['TIMESTAMP'].min()} to {df['TIMESTAMP'].max()}")
        logger.info(f"Sample of processed data for plot {plot_number}:\n{df.head().to_string()}")
        logger.info(f"Columns in processed data: {df.columns.tolist()}")
    
    return processed_data

def resample_hourly(df):
    # Set TIMESTAMP as index
    df = df.set_index('TIMESTAMP')
    
    # Identify columns to resample
    columns_to_average = df.columns.drop('Rain_1m_Tot', errors='ignore')
    
    # Create a dictionary for resampling operations
    resampling_dict = {col: 'first' for col in columns_to_average}  # Changed from 'mean' to 'first'
    if 'Rain_1m_Tot' in df.columns:
        resampling_dict['Rain_1m_Tot'] = 'sum'
    
    # Resample data
    df_resampled = df.resample('1H').agg(resampling_dict)
    
    # Reset index to make TIMESTAMP a column again
    df_resampled = df_resampled.reset_index()
    
    logger.info(f"Resampled data to hourly intervals. New shape: {df_resampled.shape}")
    
    return df_resampled
