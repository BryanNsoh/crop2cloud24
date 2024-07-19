import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def process_plot_data(plot_data, weather_data):
    processed_data = {}
    
    # Convert weather data TIMESTAMP to nanosecond precision once
    weather_data['TIMESTAMP'] = pd.to_datetime(weather_data['TIMESTAMP'], utc=True).astype('datetime64[ns, UTC]')
    
    for crop_type, treatments in plot_data.items():
        processed_data[crop_type] = {}
        for treatment, plots in treatments.items():
            processed_data[crop_type][treatment] = {}
            for plot_number, df in plots.items():
                logger.debug(f"Processing {crop_type}_{treatment}_plot_{plot_number}")
                logger.debug(f"Input data type: {type(df)}")
                logger.debug(f"Input data content: {df}")
                
                # Determine actual data range
                actual_start = df['TIMESTAMP'].min()
                actual_end = df['TIMESTAMP'].max()
                logger.info(f"Actual data range for {crop_type}_{treatment}_plot_{plot_number}: {actual_start} to {actual_end}")
                
                # Remove duplicate timestamps
                df = df.sort_values('TIMESTAMP').drop_duplicates(subset='TIMESTAMP', keep='first')
                
                # Convert plot data TIMESTAMP to nanosecond precision
                df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], utc=True).astype('datetime64[ns, UTC]')
                
                # Resample to hourly intervals without interpolation
                df = resample_hourly(df)
                
                # Log timestamp dtypes for debugging
                logger.info(f"Plot data TIMESTAMP dtype: {df['TIMESTAMP'].dtype}")
                logger.info(f"Weather data TIMESTAMP dtype: {weather_data['TIMESTAMP'].dtype}")
                
                # Perform the merge
                df = pd.merge(df, weather_data, on='TIMESTAMP', how='outer')
                
                # Identify sensor data columns (all numeric columns except TIMESTAMP and weather columns)
                weather_columns = set(weather_data.columns)
                numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
                sensor_columns = [col for col in numeric_columns if col not in weather_columns and col != 'TIMESTAMP']
                
                # Replace values below -50 with null for sensor data columns
                df[sensor_columns] = df[sensor_columns].apply(lambda x: x.where(x >= -50, pd.NA))
                
                logger.info(f"Replaced values below -50 with null for sensor columns: {sensor_columns}")
                
                # Drop all-null columns except 'swsi', 'et', and 'cwsi-th1'
                cols_to_keep = ['TIMESTAMP', 'swsi', 'et', 'cwsi-th1']
                null_columns = df.columns[df.isnull().all()].tolist()
                cols_to_drop = [col for col in null_columns if col not in cols_to_keep]
                df = df.drop(columns=cols_to_drop)
                
                # Explicitly drop 'is_actual' column if it exists
                if 'is_actual' in df.columns:
                    df = df.drop(columns=['is_actual'])
                
                logger.debug(f"Processed data type: {type(df)}")
                logger.debug(f"Processed data content: {df}")
                processed_data[crop_type][treatment][plot_number] = df
                
                logger.info(f"Processed data for {crop_type}_{treatment}_plot_{plot_number}. Shape: {df.shape}")
                logger.info(f"Processed data range: {df['TIMESTAMP'].min()} to {df['TIMESTAMP'].max()}")
                logger.info(f"Sample of processed data for {crop_type}_{treatment}_plot_{plot_number}:\n{df.head().to_string()}")
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