import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def process_plot_data(plot_data):
    processed_data = {}
    for plot_number, df in plot_data.items():
        logger.info(f"Processing data for plot {plot_number}")
        
        # Remove duplicate timestamps
        df = df.sort_values('TIMESTAMP').drop_duplicates(subset='TIMESTAMP', keep='first')
        
        # Set 'TIMESTAMP' as index for interpolation
        df = df.set_index('TIMESTAMP')
        
        # Interpolate missing values
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        df[numeric_columns] = df[numeric_columns].interpolate(method='time')
        
        # Reset index to make 'TIMESTAMP' a column again
        df = df.reset_index()
        
        processed_data[plot_number] = df
        
        logger.info(f"Processed data for plot {plot_number}. Shape: {df.shape}")
        logger.info(f"Sample of processed data for plot {plot_number}:\n{df.head().to_string()}")
    
    return processed_data