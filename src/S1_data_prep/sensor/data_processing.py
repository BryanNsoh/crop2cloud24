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
        
        # Remove duplicate timestamps
        df = df.sort_values('TIMESTAMP').drop_duplicates(subset='TIMESTAMP', keep='first')
        
        # Convert plot data TIMESTAMP to nanosecond precision
        df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], utc=True).astype('datetime64[ns, UTC]')
        
        # Set 'TIMESTAMP' as index for interpolation
        df = df.set_index('TIMESTAMP')
        
        # Interpolate missing values
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        df[numeric_columns] = df[numeric_columns].interpolate(method='time')
        
        # Reset index to make 'TIMESTAMP' a column again
        df = df.reset_index()
        
        # Log timestamp dtypes for debugging
        logger.info(f"Plot data TIMESTAMP dtype: {df['TIMESTAMP'].dtype}")
        logger.info(f"Weather data TIMESTAMP dtype: {weather_data['TIMESTAMP'].dtype}")
        
        # Perform the merge
        df = pd.merge_asof(df, weather_data, on='TIMESTAMP', direction='nearest', tolerance=pd.Timedelta('1h'))
        
        processed_data[plot_number] = df
        
        logger.info(f"Processed data for plot {plot_number}. Shape: {df.shape}")
        logger.info(f"Sample of processed data for plot {plot_number}:\n{df.head().to_string()}")
    
    return processed_data