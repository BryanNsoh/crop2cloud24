import os
import yaml
from google.cloud import bigquery
from google.oauth2 import service_account
from google.api_core import exceptions
from dotenv import load_dotenv
from logger import get_logger

# Set up logging
logger = get_logger(__name__)

def load_sensor_mapping(file_path):
    try:
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        logger.error(f"Sensor mapping file not found: {file_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {e}")
        raise

def create_bigquery_client():
    load_dotenv()
    credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    
    if not credentials_path:
        logger.error("GOOGLE_APPLICATION_CREDENTIALS not set in .env file")
        raise ValueError("GOOGLE_APPLICATION_CREDENTIALS not set in .env file")
    
    try:
        credentials = service_account.Credentials.from_service_account_file(
            credentials_path,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        return bigquery.Client(credentials=credentials, project=credentials.project_id)
    except Exception as e:
        logger.error(f"Error creating BigQuery client: {e}")
        raise

def create_table_schema(sensor_ids):
    schema = [
        bigquery.SchemaField("TIMESTAMP", "TIMESTAMP", mode="REQUIRED"),
    ]
    for sensor_id in sensor_ids:
        schema.append(bigquery.SchemaField(sensor_id, "FLOAT"))
    return schema

def ensure_dataset_exists(client, dataset_id):
    dataset_ref = client.dataset(dataset_id)
    try:
        client.get_dataset(dataset_ref)
        logger.info(f"Dataset {dataset_id} already exists")
    except exceptions.NotFound:
        dataset = bigquery.Dataset(dataset_ref)
        dataset = client.create_dataset(dataset)
        logger.info(f"Created dataset {dataset_id}")

def table_exists(client, table_id):
    try:
        client.get_table(table_id)
        return True
    except exceptions.NotFound:
        return False

def create_or_update_table(client, table_id, schema):
    if not table_exists(client, table_id):
        table = bigquery.Table(table_id, schema=schema)
        table = client.create_table(table)
        logger.info(f"Created table {table_id}")
    else:
        table = client.get_table(table_id)
        table.schema = schema
        table = client.update_table(table, ["schema"])
        logger.info(f"Updated schema for existing table {table_id}")

def process_sensor_mapping(client, sensor_mapping):
    # Group sensors by dataset_id, treatment, and collect sensor_ids
    treatment_sensor_map = {}
    for sensor in sensor_mapping:
        dataset_id = sensor['dataset_id']
        treatment = sensor['treatment']
        sensor_id = sensor['sensor_id']
        
        if dataset_id not in treatment_sensor_map:
            treatment_sensor_map[dataset_id] = {}
        if treatment not in treatment_sensor_map[dataset_id]:
            treatment_sensor_map[dataset_id][treatment] = set()
        
        treatment_sensor_map[dataset_id][treatment].add(sensor_id)
    
    # Create datasets and tables
    for dataset_id, treatments in treatment_sensor_map.items():
        ensure_dataset_exists(client, dataset_id)
        
        for treatment, sensor_ids in treatments.items():
            table_id = f"{client.project}.{dataset_id}.treatment_{treatment}"
            schema = create_table_schema(sensor_ids)
            create_or_update_table(client, table_id, schema)

def main():
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        yaml_path = os.path.join(script_dir, '..', '..', 'sensor_mapping.yaml')
        sensor_mapping = load_sensor_mapping(yaml_path)
        client = create_bigquery_client()
        process_sensor_mapping(client, sensor_mapping)
        logger.info("Process completed successfully")
    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
    
#This is a test for collapse xture