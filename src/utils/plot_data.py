import os
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import pytz
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get the project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Set up directories
DB_PATH = os.path.join(PROJECT_ROOT, 'mpc_data.db')
PLOTS_DIR = os.path.join(PROJECT_ROOT, 'plots')
HTML_PLOTS_DIR = os.path.join(PROJECT_ROOT, 'html_plots')

# Ensure directories exist
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(HTML_PLOTS_DIR, exist_ok=True)

# Custom color palette
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

# Define CST timezone
CST = pytz.timezone('America/Chicago')

def get_plot_tables(conn):
    """Get all plot tables from the database."""
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'plot_%'")
    return [row[0] for row in cursor.fetchall()]

def get_weather_data(conn):
    """Retrieve weather data from the database."""
    query = """
    SELECT TIMESTAMP, Ta_2m_Avg, TaMax_2m, TaMin_2m, RH_2m_Avg, 
           Dp_2m_Avg, WndAveSpd_3m, Solar_2m_Avg, Rain_1m_Tot
    FROM weather_data
    WHERE TIMESTAMP IS NOT NULL
    ORDER BY TIMESTAMP
    """
    df = pd.read_sql_query(query, conn, parse_dates=['TIMESTAMP'])
    
    # Convert TIMESTAMP to datetime, interpret as UTC, then convert to CST
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], utc=True)
    df['TIMESTAMP'] = df['TIMESTAMP'].dt.tz_convert(CST)
    df['TIMESTAMP'] = df['TIMESTAMP'] - pd.Timedelta(hours=5)
    
    ###
    # SOMETHING IS FUCKING WRONG HERE WHY I GOTTA DECREMENT THE THING TWICE FOR IT TO WORK??? WHAT IS HTE ROOT CAUSE OF THIS? ANYWAYS AT LEASE WE GETTIN SOMEWHERE
    
    
    # Remove timezone information for easier plotting
    df['TIMESTAMP'] = df['TIMESTAMP'].dt.tz_localize(None)
    
    df = df.dropna(subset=['TIMESTAMP'])
    
    # Replace negative values with NaN for relevant columns
    numeric_columns = ['Ta_2m_Avg', 'TaMax_2m', 'TaMin_2m', 'RH_2m_Avg', 'WndAveSpd_3m', 'Solar_2m_Avg', 'Rain_1m_Tot']
    for col in numeric_columns:
        df[col] = df[col].where(df[col] >= 0)
    
    logger.info(f"Weather data retrieved. Shape: {df.shape}")
    logger.info(f"Weather data range: {df['TIMESTAMP'].min()} to {df['TIMESTAMP'].max()} (CST)")
    return df

def clean_data(conn, table_name):
    """Clean and prepare data for a specific plot."""
    query = f"""
    SELECT *
    FROM {table_name}
    WHERE TIMESTAMP IS NOT NULL AND is_actual = 1
    ORDER BY TIMESTAMP
    """
    
    df = pd.read_sql_query(query, conn, parse_dates=['TIMESTAMP'])
    
    # Convert TIMESTAMP to datetime, interpret as UTC, then convert to CST
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], utc=True)
    df['TIMESTAMP'] = df['TIMESTAMP'].dt.tz_convert(CST)
    
    # Remove timezone information for easier plotting
    df['TIMESTAMP'] = df['TIMESTAMP'].dt.tz_localize(None)
    
    df = df.dropna(subset=['TIMESTAMP'])
    
    # Replace negative values with NaN for numeric columns
    numeric_columns = df.select_dtypes(include=['float64']).columns
    for col in numeric_columns:
        df[col] = df[col].where(df[col] >= 0)
    
    # Replace VWC values of 0 with NaN
    tdr_columns = [col for col in df.columns if col.startswith('TDR') and not col.endswith('_pred')]
    df[tdr_columns] = df[tdr_columns].replace(0, np.nan)
    
    logger.info(f"Data cleaned for {table_name}. Shape: {df.shape}")
    logger.info(f"Date range: {df['TIMESTAMP'].min()} to {df['TIMESTAMP'].max()} (CST)")
    
    return df

def log_data_summary(df, plot_number):
    """Log summary statistics for the dataframe."""
    logger.info(f"Data summary for plot {plot_number}:")
    logger.info(f"Date range: {df['TIMESTAMP'].min()} to {df['TIMESTAMP'].max()}")
    
    for column in df.columns:
        if column != 'TIMESTAMP' and column != 'is_actual':
            non_null_data = df[column].dropna()
            if not non_null_data.empty:
                logger.info(f"{column}:")
                logger.info(f"  Range: {non_null_data.min():.2f} to {non_null_data.max():.2f}")
                logger.info(f"  Mean: {non_null_data.mean():.2f}")
                logger.info(f"  Median: {non_null_data.median():.2f}")
                logger.info(f"  Non-null count: {non_null_data.count()} out of {len(df)}")
            else:
                logger.warning(f"{column}: All values are null")

def create_static_plot(df, weather_df, plot_number):
    """Create a static plot using matplotlib."""
    fig, axs = plt.subplots(3, 2, figsize=(20, 30), sharex=True)
    fig.suptitle(f'Data for Plot {plot_number}', fontsize=16)

    # VWC Plot
    tdr_columns = [col for col in df.columns if col.startswith('TDR') and not col.endswith('_pred')]
    for i, col in enumerate(tdr_columns):
        axs[0, 0].plot(df['TIMESTAMP'], df[col], label=col, color=COLORS[i % len(COLORS)])
    axs[0, 0].set_ylabel('VWC (%)')
    axs[0, 0].legend()

    # Canopy Temperature and Air Temperature Plot
    irt_column = next((col for col in df.columns if col.startswith('IRT') and not col.endswith('_pred')), None)
    if irt_column:
        axs[1, 0].plot(df['TIMESTAMP'], df[irt_column], label='Canopy Temp', color=COLORS[0])
        logger.info(f"Plotting canopy temperature from column: {irt_column}")
    else:
        logger.warning("No IRT column found for canopy temperature")
    if 'Ta_2m_Avg' in weather_df.columns:
        axs[1, 0].plot(weather_df['TIMESTAMP'], weather_df['Ta_2m_Avg'], label='Air Temp', color=COLORS[1])
        logger.info("Plotting air temperature from Ta_2m_Avg column")
    else:
        logger.warning("Ta_2m_Avg column not found in weather dataframe")
    axs[1, 0].set_ylabel('Temperature (°C)')
    axs[1, 0].legend()

    # Precipitation Plot
    if 'Rain_1m_Tot' in weather_df.columns:
        valid_rain = weather_df.dropna(subset=['Rain_1m_Tot'])
        axs[2, 0].bar(valid_rain['TIMESTAMP'], valid_rain['Rain_1m_Tot'], label='Precipitation', color=COLORS[2])
        axs[2, 0].set_ylabel('Precipitation (mm)')
        axs[2, 0].legend()
        logger.info(f"Plotting precipitation data. Non-null count: {len(valid_rain)}")
    else:
        logger.warning("Rain_1m_Tot column not found in weather dataframe")

    # CWSI and SWSI Plot
    if 'cwsi' in df.columns and 'swsi' in df.columns:
        axs[0, 1].plot(df['TIMESTAMP'], df['cwsi'], label='CWSI', color=COLORS[2])
        axs[0, 1].plot(df['TIMESTAMP'], df['swsi'], label='SWSI', color=COLORS[3])
        axs[0, 1].set_ylabel('Index')
        axs[0, 1].legend()
        logger.info("Plotting CWSI and SWSI data")
    else:
        logger.warning("CWSI or SWSI column not found in dataframe")

    # ET Plot
    if 'et' in df.columns:
        axs[1, 1].plot(df['TIMESTAMP'], df['et'], label='ET', color=COLORS[4])
        axs[1, 1].set_ylabel('ET (mm)')
        axs[1, 1].legend()
        logger.info("Plotting ET data")
    else:
        logger.warning("ET column not found in dataframe")

    # Relative Humidity Plot
    if 'RH_2m_Avg' in weather_df.columns:
        axs[2, 1].plot(weather_df['TIMESTAMP'], weather_df['RH_2m_Avg'], label='RH', color=COLORS[1])
        axs[2, 1].set_ylabel('Relative Humidity (%)')
        axs[2, 1].legend()
        logger.info("Plotting Relative Humidity data")
    else:
        logger.warning("RH_2m_Avg column not found in weather dataframe")

    # Format x-axis
    for ax in axs.flat:
        ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d %H:%M', tz=CST))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'{plot_number}_static.png'))
    plt.close()
    logger.info(f"Static plot saved for plot {plot_number}")

def create_interactive_plot(df, weather_df, plot_number):
    """Create an interactive plot using plotly."""
    fig = make_subplots(rows=3, cols=2, shared_xaxes=True, shared_yaxes=False)

    # VWC Plot
    tdr_columns = [col for col in df.columns if col.startswith('TDR') and not col.endswith('_pred')]
    for i, col in enumerate(tdr_columns):
        fig.add_trace(go.Scatter(x=df['TIMESTAMP'], y=df[col], name=col, line=dict(color=COLORS[i % len(COLORS)])),
                      row=1, col=1)

    # Canopy Temperature and Air Temperature Plot
    irt_column = next((col for col in df.columns if col.startswith('IRT') and not col.endswith('_pred')), None)
    if irt_column:
        fig.add_trace(go.Scatter(x=df['TIMESTAMP'], y=df[irt_column], name='Canopy Temp', line=dict(color=COLORS[0])),
                      row=2, col=1)
        logger.info(f"Plotting canopy temperature from column: {irt_column}")
    else:
        logger.warning("No IRT column found for canopy temperature")
    if 'Ta_2m_Avg' in weather_df.columns:
        fig.add_trace(go.Scatter(x=weather_df['TIMESTAMP'], y=weather_df['Ta_2m_Avg'], name='Air Temp', line=dict(color=COLORS[1])),
                      row=2, col=1)
        logger.info("Plotting air temperature from Ta_2m_Avg column")
    else:
        logger.warning("Ta_2m_Avg column not found in weather dataframe")

    # Precipitation Plot
    if 'Rain_1m_Tot' in weather_df.columns:
        valid_rain = weather_df.dropna(subset=['Rain_1m_Tot'])
        fig.add_trace(go.Bar(x=valid_rain['TIMESTAMP'], y=valid_rain['Rain_1m_Tot'], name='Precipitation', marker_color=COLORS[2]),
                      row=3, col=1)
        logger.info(f"Plotting precipitation data. Non-null count: {len(valid_rain)}")
    else:
        logger.warning("Rain_1m_Tot column not found in weather dataframe")

    # CWSI and SWSI Plot
    if 'cwsi' in df.columns and 'swsi' in df.columns:
        fig.add_trace(go.Scatter(x=df['TIMESTAMP'], y=df['cwsi'], name='CWSI', line=dict(color=COLORS[2])),
                      row=1, col=2)
        fig.add_trace(go.Scatter(x=df['TIMESTAMP'], y=df['swsi'], name='SWSI', line=dict(color=COLORS[3])),
                      row=1, col=2)
        logger.info("Plotting CWSI and SWSI data")
    else:
        logger.warning("CWSI or SWSI column not found in dataframe")

    # ET Plot
    if 'et' in df.columns:
        fig.add_trace(go.Scatter(x=df['TIMESTAMP'], y=df['et'], name='ET', line=dict(color=COLORS[4])),
                      row=2, col=2)
        logger.info("Plotting ET data")
    else:
        logger.warning("ET column not found in dataframe")

    # Relative Humidity Plot
    if 'RH_2m_Avg' in weather_df.columns:
        fig.add_trace(go.Scatter(x=weather_df['TIMESTAMP'], y=weather_df['RH_2m_Avg'], name='RH', line=dict(color=COLORS[1])),
                      row=3, col=2)
        logger.info("Plotting Relative Humidity data")
    else:
        logger.warning("RH_2m_Avg column not found in weather dataframe")

    fig.update_layout(title=f'Data for Plot {plot_number}', height=900, width=1200)

    # Update y-axis titles
    fig.update_yaxes(title_text="VWC (%)", row=1, col=1)
    fig.update_yaxes(title_text="Temperature (°C)", row=2, col=1)
    fig.update_yaxes(title_text="Precipitation (mm)", row=3, col=1)
    fig.update_yaxes(title_text="Index", row=1, col=2)
    fig.update_yaxes(title_text="ET (mm)", row=2, col=2)
    fig.update_yaxes(title_text="Relative Humidity (%)", row=3, col=2)

    # Update x-axis to show CST time
    fig.update_xaxes(tickformat="%Y-%m-%d %H:%M")

    fig.write_html(os.path.join(HTML_PLOTS_DIR, f'{plot_number}_interactive.html'))
    logger.info(f"Interactive plot saved for plot {plot_number}")

def generate_plots(plot_numbers=None):
    """
    Generate plots for specified plot numbers or all plots if none specified.
    
    :param plot_numbers: List of plot numbers to generate plots for. If None, generates for all plots.
    """
    conn = sqlite3.connect(DB_PATH)
    all_plot_tables = get_plot_tables(conn)
    
    if plot_numbers is None:
        plot_tables = all_plot_tables
    else:
        plot_tables = [f"plot_{num}" for num in plot_numbers if f"plot_{num}" in all_plot_tables]

    # Get weather data
    weather_df = get_weather_data(conn)
    
    for table in plot_tables:
        plot_number = table.split('_')[1]
        logger.info(f"Processing data for plot {plot_number}")
        
        df = clean_data(conn, table)
        
        if not df.empty:
            log_data_summary(df, plot_number)
            
            # Filter weather data to match the plot data's date range
            plot_start = df['TIMESTAMP'].min()
            plot_end = df['TIMESTAMP'].max()
            filtered_weather_df = weather_df[(weather_df['TIMESTAMP'] >= plot_start) & (weather_df['TIMESTAMP'] <= plot_end)]
            
            create_static_plot(df, filtered_weather_df, plot_number)
            create_interactive_plot(df, filtered_weather_df, plot_number)
            logger.info(f"Generated plots for {table}")
        else:
            logger.warning(f"No data available for {table}")

    conn.close()
    logger.info("All plots generated successfully.")

if __name__ == "__main__":
    generate_plots()