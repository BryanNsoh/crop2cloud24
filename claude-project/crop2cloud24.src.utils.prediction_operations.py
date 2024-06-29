import pandas as pd
from google.cloud import bigquery
from logger import get_logger
from bigquery_operations import create_bigquery_client, insert_or_update_data

logger = get_logger(__name__)

def insert_predictions(client, table_id, predictions_df):
    """
    Insert prediction data into BigQuery table.
    
    :param client: BigQuery client
    :param table_id: Full table ID (project.dataset.table)
    :param predictions_df: DataFrame containing predictions
    """
    # Ensure prediction_timestamp is set
    predictions_df['prediction_timestamp'] = pd.Timestamp.now(tz='UTC')
    
    # Insert predictions
    insert_or_update_data(client, table_id, predictions_df, is_actual=False)

def get_data_for_analysis(client, table_id, start_time, end_time):
    """
    Retrieve both actual and predicted data for analysis.
    
    :param client: BigQuery client
    :param table_id: Full table ID (project.dataset.table)
    :param start_time: Start of the time range
    :param end_time: End of the time range
    :return: DataFrame containing both actual and predicted data
    """
    query = f"""
    SELECT *
    FROM `{table_id}`
    WHERE TIMESTAMP BETWEEN '{start_time}' AND '{end_time}'
    ORDER BY TIMESTAMP, is_actual DESC
    """
    
    query_job = client.query(query)
    results = query_job.result()
    
    df = results.to_dataframe()
    
    # Process the dataframe to combine actual and predicted values
    df_actual = df[df['is_actual']].set_index('TIMESTAMP')
    df_predicted = df[~df['is_actual']].set_index('TIMESTAMP')
    
    # Combine actual and predicted data
    df_combined = pd.concat([df_actual, df_predicted], axis=1, keys=['actual', 'predicted'])
    df_combined.columns = ['_'.join(col).strip() for col in df_combined.columns.values]
    
    return df_combined

def get_latest_data_for_prediction(client, table_id, hours=24):
    """
    Retrieve the latest actual data for making predictions.
    
    :param client: BigQuery client
    :param table_id: Full table ID (project.dataset.table)
    :param hours: Number of hours of data to retrieve
    :return: DataFrame containing the latest actual data
    """
    query = f"""
    SELECT *
    FROM `{table_id}`
    WHERE is_actual = TRUE
    ORDER BY TIMESTAMP DESC
    LIMIT {hours}
    """
    
    query_job = client.query(query)
    results = query_job.result()
    
    df = results.to_dataframe()
    df = df.sort_values('TIMESTAMP')
    
    return df