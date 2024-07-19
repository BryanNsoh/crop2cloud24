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
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

# Define CST timezone
CST = pytz.timezone('America/Chicago')

class PlotGenerator:
    def __init__(self, db_path, plot_numbers):
        self.db_path = db_path
        self.plot_numbers = plot_numbers
        self.conn = sqlite3.connect(self.db_path)

    def get_plot_tables(self):
        return [f"LINEAR_CORN_trt1_plot_{num}" for num in self.plot_numbers]

    def clean_data(self, table_name):
        query = f"""
        SELECT *
        FROM {table_name}
        WHERE TIMESTAMP IS NOT NULL
        ORDER BY TIMESTAMP
        """
        
        df = pd.read_sql_query(query, self.conn, parse_dates=['TIMESTAMP'])
        df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], utc=True).dt.tz_convert(CST)
        df = df.dropna(subset=['TIMESTAMP'])
        
        numeric_columns = df.select_dtypes(include=['float64']).columns
        for col in numeric_columns:
            df[col] = df[col].where(df[col] >= 0)
        
        tdr_columns = [col for col in df.columns if col.startswith('TDR') and not col.endswith('_pred')]
        df[tdr_columns] = df[tdr_columns].replace(0, np.nan)
        
        df_cwsi = df[df['TIMESTAMP'].dt.hour.between(12, 16)]
        
        logger.info(f"Data cleaned for {table_name}. Shape: {df.shape}")
        logger.info(f"Date range: {df['TIMESTAMP'].min()} to {df['TIMESTAMP'].max()} (CST)")
        logger.info(f"CWSI data filtered. Shape: {df_cwsi.shape}")
        
        return df, df_cwsi

    def log_data_summary(self, df, plot_number):
        logger.info(f"Data summary for plot {plot_number}:")
        logger.info(f"Date range: {df['TIMESTAMP'].min()} to {df['TIMESTAMP'].max()}")
        
        plot_columns = ['TIMESTAMP', 'Ta_2m_Avg', 'Rain_1m_Tot', 'et', 'RH_2m_Avg']
        plot_columns += [col for col in df.columns if col.startswith('TDR') and not col.endswith('_pred')]
        plot_columns += ['IRT', 'cwsi_th1']

        for column in plot_columns:
            if column in df.columns:
                non_null_data = df[column].dropna()
                if not non_null_data.empty:
                    logger.info(f"{column}:")
                    logger.info(f"  Range: {non_null_data.min():.2f} to {non_null_data.max():.2f}")
                    logger.info(f"  Mean: {non_null_data.mean():.2f}")
                    logger.info(f"  Median: {non_null_data.median():.2f}")
                    logger.info(f"  Non-null count: {non_null_data.count()} out of {len(df)}")
                else:
                    logger.warning(f"{column}: All values are null")
            else:
                logger.warning(f"{column} column not found in dataframe")

    def create_static_plot(self, df, df_cwsi, plot_number):
        fig, axs = plt.subplots(3, 2, figsize=(20, 30), sharex=True)
        fig.suptitle(f'Data for Plot {plot_number}', fontsize=16)

        self._plot_vwc(axs[0, 0], df)
        self._plot_temperature(axs[1, 0], df)
        self._plot_precipitation(axs[2, 0], df)
        self._plot_cwsi(axs[0, 1], df_cwsi)
        self._plot_et(axs[1, 1], df)
        self._plot_humidity(axs[2, 1], df)

        for ax in axs.flat:
            ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d %H:%M', tz=CST))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f'{plot_number}_static.png'))
        plt.close()
        logger.info(f"Static plot saved for plot {plot_number}")

    def create_interactive_plot(self, df, df_cwsi, plot_number):
        fig = make_subplots(rows=3, cols=2, shared_xaxes=True, shared_yaxes=False)

        self._add_vwc_trace(fig, df, row=1, col=1)
        self._add_temperature_trace(fig, df, row=2, col=1)
        self._add_precipitation_trace(fig, df, row=3, col=1)
        self._add_cwsi_trace(fig, df_cwsi, row=1, col=2)
        self._add_et_trace(fig, df, row=2, col=2)
        self._add_humidity_trace(fig, df, row=3, col=2)

        fig.update_layout(title=f'Data for Plot {plot_number}', height=900, width=1200)

        fig.update_yaxes(title_text="VWC (%)", row=1, col=1)
        fig.update_yaxes(title_text="Temperature (°C)", row=2, col=1)
        fig.update_yaxes(title_text="Precipitation (mm)", row=3, col=1)
        fig.update_yaxes(title_text="Index", row=1, col=2)
        fig.update_yaxes(title_text="ET (mm)", row=2, col=2)
        fig.update_yaxes(title_text="Relative Humidity (%)", row=3, col=2)

        fig.update_xaxes(tickformat="%Y-%m-%d %H:%M")

        fig.write_html(os.path.join(HTML_PLOTS_DIR, f'{plot_number}_interactive.html'))
        logger.info(f"Interactive plot saved for plot {plot_number}")

    def _plot_vwc(self, ax, df):
        tdr_columns = [col for col in df.columns if col.startswith('TDR') and not col.endswith('_pred')]
        for i, col in enumerate(tdr_columns):
            ax.plot(df['TIMESTAMP'], df[col], label=col, color=COLORS[i % len(COLORS)])
        ax.set_ylabel('VWC (%)')
        ax.legend()

    def _plot_temperature(self, ax, df):
        irt_column = next((col for col in df.columns if col.startswith('IRT') and not col.endswith('_pred')), None)
        if irt_column:
            ax.plot(df['TIMESTAMP'], df[irt_column], label='Canopy Temp', color=COLORS[0])
            logger.info(f"Plotting canopy temperature from column: {irt_column}")
        else:
            logger.warning("No IRT column found for canopy temperature")
        if 'Ta_2m_Avg' in df.columns:
            ax.plot(df['TIMESTAMP'], df['Ta_2m_Avg'], label='Air Temp', color=COLORS[1])
            logger.info("Plotting air temperature from Ta_2m_Avg column")
        else:
            logger.warning("Ta_2m_Avg column not found in dataframe")
        ax.set_ylabel('Temperature (°C)')
        ax.legend()

    def _plot_precipitation(self, ax, df):
        if 'Rain_1m_Tot' in df.columns:
            ax.bar(df['TIMESTAMP'], df['Rain_1m_Tot'], label='Precipitation', color=COLORS[2])
            ax.set_ylabel('Precipitation (mm)')
            ax.legend()
            logger.info("Plotting precipitation data")
        else:
            logger.warning("Rain_1m_Tot column not found in dataframe")

    def _plot_cwsi(self, ax, df_cwsi):
        if 'cwsi_th1' in df_cwsi.columns:
            ax.plot(df_cwsi['TIMESTAMP'], df_cwsi['cwsi_th1'], label='CWSI-TH1', color=COLORS[3])
            ax.set_ylabel('Index')
            ax.legend()
            logger.info("Plotting CWSI-TH1 data (12 PM - 5 PM CST)")
        else:
            logger.warning("cwsi_th1 column not found in dataframe")

    def _plot_et(self, ax, df):
        if 'et' in df.columns:
            ax.plot(df['TIMESTAMP'], df['et'], label='ET', color=COLORS[4])
            ax.set_ylabel('ET (mm)')
            ax.legend()
            logger.info("Plotting ET data")
        else:
            logger.warning("ET column not found in dataframe")

    def _plot_humidity(self, ax, df):
        if 'RH_2m_Avg' in df.columns:
            ax.plot(df['TIMESTAMP'], df['RH_2m_Avg'], label='RH', color=COLORS[1])
            ax.set_ylabel('Relative Humidity (%)')
            ax.legend()
            logger.info("Plotting Relative Humidity data")
        else:
            logger.warning("RH_2m_Avg column not found in dataframe")

    def _add_vwc_trace(self, fig, df, row, col):
        tdr_columns = [col for col in df.columns if col.startswith('TDR') and not col.endswith('_pred')]
        for i, tdr_col in enumerate(tdr_columns):
            fig.add_trace(go.Scatter(x=df['TIMESTAMP'], y=df[tdr_col], name=tdr_col, line=dict(color=COLORS[i % len(COLORS)])),
                          row=row, col=col)

    def _add_temperature_trace(self, fig, df, row, col):
        irt_column = next((col for col in df.columns if col.startswith('IRT') and not col.endswith('_pred')), None)
        if irt_column:
            fig.add_trace(go.Scatter(x=df['TIMESTAMP'], y=df[irt_column], name='Canopy Temp', line=dict(color=COLORS[0])),
                          row=row, col=col)
            logger.info(f"Plotting canopy temperature from column: {irt_column}")
        else:
            logger.warning("No IRT column found for canopy temperature")
        if 'Ta_2m_Avg' in df.columns:
            fig.add_trace(go.Scatter(x=df['TIMESTAMP'], y=df['Ta_2m_Avg'], name='Air Temp', line=dict(color=COLORS[1])),
                          row=row, col=col)
            logger.info("Plotting air temperature from Ta_2m_Avg column")
        else:
            logger.warning("Ta_2m_Avg column not found in dataframe")

    def _add_precipitation_trace(self, fig, df, row, col):
        if 'Rain_1m_Tot' in df.columns:
            fig.add_trace(go.Bar(x=df['TIMESTAMP'], y=df['Rain_1m_Tot'], name='Precipitation', marker_color=COLORS[2]),
                          row=row, col=col)
            logger.info("Plotting precipitation data")
        else:
            logger.warning("Rain_1m_Tot column not found in dataframe")

    def _add_cwsi_trace(self, fig, df_cwsi, row, col):
        if 'cwsi_th1' in df_cwsi.columns:
            fig.add_trace(go.Scatter(x=df_cwsi['TIMESTAMP'], y=df_cwsi['cwsi_th1'], name='CWSI-TH1', line=dict(color=COLORS[3])),
                          row=row, col=col)
            logger.info("Plotting CWSI-TH1 data (12 PM - 5 PM CST)")
        else:
            logger.warning("cwsi_th1 column not found in dataframe")

    def _add_et_trace(self, fig, df, row, col):
        if 'et' in df.columns:
            fig.add_trace(go.Scatter(x=df['TIMESTAMP'], y=df['et'], name='ET', line=dict(color=COLORS[4])),
                          row=row, col=col)
            logger.info("Plotting ET data")
        else:
            logger.warning("ET column not found in dataframe")

    def _add_humidity_trace(self, fig, df, row, col):
        if 'RH_2m_Avg' in df.columns:
            fig.add_trace(go.Scatter(x=df['TIMESTAMP'], y=df['RH_2m_Avg'], name='RH', line=dict(color=COLORS[1])),
                          row=row, col=col)
            logger.info("Plotting Relative Humidity data")
        else:
            logger.warning("RH_2m_Avg column not found in dataframe")

    def generate_plots(self):
        plot_tables = self.get_plot_tables()
        
        for table in plot_tables:
            plot_number = table.split('_')[-1]
            logger.info(f"Processing data for plot {plot_number}")
            
            df, df_cwsi = self.clean_data(table)
            
            if not df.empty:
                self.log_data_summary(df, plot_number)
                self.create_static_plot(df, df_cwsi, plot_number)
                self.create_interactive_plot(df, df_cwsi, plot_number)
                logger.info(f"Generated plots for {table}")
            else:
                logger.warning(f"No data available for {table}")

        self.conn.close()
        logger.info("All plots generated successfully.")

def main():
    plot_numbers = [5006, 5010, 5023]  # You can modify this list as needed
    plot_generator = PlotGenerator(DB_PATH, plot_numbers)
    plot_generator.generate_plots()

if __name__ == "__main__":
    main()
                