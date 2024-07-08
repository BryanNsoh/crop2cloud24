# src/utils/plot_data.py

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

def get_plot_tables(conn):
    """Get all plot tables from the database."""
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'plot_%'")
    return [row[0] for row in cursor.fetchall()]

def clean_data(conn, table_name, start_date, end_date):
    """Clean and prepare data for a specific plot."""
    query = f"""
    SELECT p.*, w.Ta_2m_Avg, w.RH_2m_Avg, w.Rain_1m_Tot
    FROM {table_name} p
    LEFT JOIN weather_data w ON p.TIMESTAMP = w.TIMESTAMP
    WHERE p.TIMESTAMP BETWEEN '{start_date}' AND '{end_date}'
    AND p.is_actual = 1
    ORDER BY p.TIMESTAMP
    """
    df = pd.read_sql_query(query, conn)
    
    # Convert TIMESTAMP to datetime using ISO8601 format
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], format='ISO8601')
    
    # Replace negative values with NaN
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].where(df[col] >= 0)
    
    # Replace VWC values of 0 with NaN
    tdr_columns = [col for col in df.columns if col.startswith('TDR') and not col.endswith('_pred')]
    df[tdr_columns] = df[tdr_columns].replace(0, np.nan)
    
    # Convert None to NaN for precipitation data
    df['Rain_1m_Tot'] = pd.to_numeric(df['Rain_1m_Tot'], errors='coerce')
    
    return df

def create_static_plot(df, plot_number):
    """Create a static plot using matplotlib."""
    fig, axs = plt.subplots(3, 2, figsize=(20, 30), sharex=True)
    fig.suptitle(f'Data for Plot {plot_number}', fontsize=16)

    # VWC Plot
    tdr_columns = [col for col in df.columns if col.startswith('TDR') and not col.endswith('_pred')]
    for i, col in enumerate(tdr_columns):
        axs[0, 0].plot(df['TIMESTAMP'], df[col], label=col, color=COLORS[i % len(COLORS)])
    axs[0, 0].set_ylabel('VWC (%)')
    axs[0, 0].legend()

    # Canopy Temperature Plot
    irt_column = next((col for col in df.columns if col.startswith('IRT') and not col.endswith('_pred')), None)
    if irt_column:
        axs[1, 0].plot(df['TIMESTAMP'], df[irt_column], label='Canopy Temp', color=COLORS[0])
    if 'Ta_2m_Avg' in df.columns:
        axs[1, 0].plot(df['TIMESTAMP'], df['Ta_2m_Avg'], label='Air Temp', color=COLORS[1])
    axs[1, 0].set_ylabel('Temperature (°C)')
    axs[1, 0].legend()

    # Precipitation Plot
    if 'Rain_1m_Tot' in df.columns:
        valid_rain = df.dropna(subset=['Rain_1m_Tot'])
        axs[2, 0].bar(valid_rain['TIMESTAMP'], valid_rain['Rain_1m_Tot'], label='Precipitation', color=COLORS[2])
        axs[2, 0].set_ylabel('Precipitation (mm)')
        axs[2, 0].legend()

    # CWSI and SWSI Plot
    if 'cwsi' in df.columns and 'swsi' in df.columns:
        axs[0, 1].plot(df['TIMESTAMP'], df['cwsi'], label='CWSI', color=COLORS[2])
        axs[0, 1].plot(df['TIMESTAMP'], df['swsi'], label='SWSI', color=COLORS[3])
        axs[0, 1].set_ylabel('Index')
        axs[0, 1].legend()

    # ET Plot
    if 'et' in df.columns:
        axs[1, 1].plot(df['TIMESTAMP'], df['et'], label='ET', color=COLORS[4])
        axs[1, 1].set_ylabel('ET (mm)')
        axs[1, 1].legend()

    # Relative Humidity Plot
    if 'RH_2m_Avg' in df.columns:
        axs[2, 1].plot(df['TIMESTAMP'], df['RH_2m_Avg'], label='RH', color=COLORS[1])
        axs[2, 1].set_ylabel('Relative Humidity (%)')
        axs[2, 1].legend()

    # Format x-axis
    for ax in axs.flat:
        ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'{plot_number}_static.png'))
    plt.close()

def create_interactive_plot(df, plot_number):
    """Create an interactive plot using plotly."""
    fig = make_subplots(rows=3, cols=2, shared_xaxes=True, shared_yaxes=False)

    # VWC Plot
    tdr_columns = [col for col in df.columns if col.startswith('TDR') and not col.endswith('_pred')]
    for i, col in enumerate(tdr_columns):
        fig.add_trace(go.Scatter(x=df['TIMESTAMP'], y=df[col], name=col, line=dict(color=COLORS[i % len(COLORS)])),
                      row=1, col=1)

    # Canopy Temperature Plot
    irt_column = next((col for col in df.columns if col.startswith('IRT') and not col.endswith('_pred')), None)
    if irt_column:
        fig.add_trace(go.Scatter(x=df['TIMESTAMP'], y=df[irt_column], name='Canopy Temp', line=dict(color=COLORS[0])),
                      row=2, col=1)
    if 'Ta_2m_Avg' in df.columns:
        fig.add_trace(go.Scatter(x=df['TIMESTAMP'], y=df['Ta_2m_Avg'], name='Air Temp', line=dict(color=COLORS[1])),
                      row=2, col=1)

    # Precipitation Plot
    if 'Rain_1m_Tot' in df.columns:
        valid_rain = df.dropna(subset=['Rain_1m_Tot'])
        fig.add_trace(go.Bar(x=valid_rain['TIMESTAMP'], y=valid_rain['Rain_1m_Tot'], name='Precipitation', marker_color=COLORS[2]),
                      row=3, col=1)

    # CWSI and SWSI Plot
    if 'cwsi' in df.columns and 'swsi' in df.columns:
        fig.add_trace(go.Scatter(x=df['TIMESTAMP'], y=df['cwsi'], name='CWSI', line=dict(color=COLORS[2])),
                      row=1, col=2)
        fig.add_trace(go.Scatter(x=df['TIMESTAMP'], y=df['swsi'], name='SWSI', line=dict(color=COLORS[3])),
                      row=1, col=2)

    # ET Plot
    if 'et' in df.columns:
        fig.add_trace(go.Scatter(x=df['TIMESTAMP'], y=df['et'], name='ET', line=dict(color=COLORS[4])),
                      row=2, col=2)

    # Relative Humidity Plot
    if 'RH_2m_Avg' in df.columns:
        fig.add_trace(go.Scatter(x=df['TIMESTAMP'], y=df['RH_2m_Avg'], name='RH', line=dict(color=COLORS[1])),
                      row=3, col=2)

    fig.update_layout(title=f'Data for Plot {plot_number}', height=900, width=1200)

    # Update y-axis titles
    fig.update_yaxes(title_text="VWC (%)", row=1, col=1)
    fig.update_yaxes(title_text="Temperature (°C)", row=2, col=1)
    fig.update_yaxes(title_text="Precipitation (mm)", row=3, col=1)
    fig.update_yaxes(title_text="Index", row=1, col=2)
    fig.update_yaxes(title_text="ET (mm)", row=2, col=2)
    fig.update_yaxes(title_text="Relative Humidity (%)", row=3, col=2)

    fig.write_html(os.path.join(HTML_PLOTS_DIR, f'{plot_number}_interactive.html'))

def generate_plots(plot_numbers=None, days=None):
    """
    Generate plots for specified plot numbers or all plots if none specified.
    
    :param plot_numbers: List of plot numbers to generate plots for. If None, generates for all plots.
    :param days: Number of days of data to include in the plots. If None, includes all available data.
    """
    conn = sqlite3.connect(DB_PATH)
    all_plot_tables = get_plot_tables(conn)
    
    if plot_numbers is None:
        plot_tables = all_plot_tables
    else:
        plot_tables = [f"plot_{num}" for num in plot_numbers if f"plot_{num}" in all_plot_tables]

    end_date = datetime.now(pytz.UTC)
    
    for table in plot_tables:
        plot_number = table.split('_')[1]
        
        if days is None:
            # If days is None, fetch all available data
            query = f"""
            SELECT MIN(TIMESTAMP) as start_date
            FROM {table}
            """
            cursor = conn.cursor()
            cursor.execute(query)
            result = cursor.fetchone()
            start_date = pd.to_datetime(result[0], utc=True) if result[0] else end_date
        else:
            start_date = end_date - timedelta(days=days)

        df = clean_data(conn, table, start_date, end_date)
        
        if not df.empty:
            create_static_plot(df, plot_number)
            create_interactive_plot(df, plot_number)
            print(f"Generated plots for {table}")
        else:
            print(f"No data available for {table}")

    conn.close()
    print("All plots generated successfully.")

if __name__ == "__main__":
    generate_plots()