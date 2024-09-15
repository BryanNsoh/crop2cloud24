import os
import yaml
from google.cloud import bigquery
from google.oauth2 import service_account
from google.api_core import exceptions
from dotenv import load_dotenv
from logger import get_logger

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
        schema.append(bigquery.SchemaField(sensor_id, "FLOAT64", mode="NULLABLE"))
    
    # Add stress indices columns
    for index in ['cwsi', 'et', 'swsi']:
        schema.append(bigquery.SchemaField(index, "FLOAT64", mode="NULLABLE"))
    
    return schema

def ensure_dataset_exists(client, dataset_id, delete_existing=False):
    dataset_ref = client.dataset(dataset_id)
    try:
        if delete_existing:
            client.delete_dataset(dataset_ref, delete_contents=True, not_found_ok=True)
            logger.info(f"Deleted existing dataset {dataset_id}")
        dataset = client.get_dataset(dataset_ref)
        logger.info(f"Dataset {dataset_id} already exists")
    except exceptions.NotFound:
        dataset = bigquery.Dataset(dataset_ref)
        dataset = client.create_dataset(dataset)
        logger.info(f"Created dataset {dataset_id}")

def update_table_schema(client, table_id, new_schema):
    table = client.get_table(table_id)
    existing_fields = set(field.name for field in table.schema)
    new_fields = set(field.name for field in new_schema)
    
    fields_to_add = new_fields - existing_fields
    
    if fields_to_add:
        updated_schema = table.schema[:]
        for field_name in fields_to_add:
            new_field = next(field for field in new_schema if field.name == field_name)
            updated_schema.append(new_field)
        
        table.schema = updated_schema
        table = client.update_table(table, ["schema"])
        logger.info(f"Updated schema for table {table_id}. Added fields: {fields_to_add}")
    else:
        logger.info(f"No new fields to add for table {table_id}")

def create_or_update_table(client, table_id, schema, delete_existing=False):
    try:
        if delete_existing:
            client.delete_table(table_id, not_found_ok=True)
            logger.info(f"Deleted existing table {table_id}")
        table = client.get_table(table_id)
        update_table_schema(client, table_id, schema)
    except exceptions.NotFound:
        table = bigquery.Table(table_id, schema=schema)
        table = client.create_table(table)
        logger.info(f"Created table {table_id}")

def process_sensor_mapping(client, sensor_mapping, delete_existing=False):
    field_treatment_plot_sensor_map = {}
    for sensor in sensor_mapping:
        field = sensor['field']
        treatment = sensor['treatment']
        plot_number = sensor['plot_number']
        sensor_id = sensor['sensor_id']
        
        if field not in field_treatment_plot_sensor_map:
            field_treatment_plot_sensor_map[field] = {}
        if treatment not in field_treatment_plot_sensor_map[field]:
            field_treatment_plot_sensor_map[field][treatment] = {}
        if plot_number not in field_treatment_plot_sensor_map[field][treatment]:
            field_treatment_plot_sensor_map[field][treatment][plot_number] = set()
        
        field_treatment_plot_sensor_map[field][treatment][plot_number].add(sensor_id)
    
    for field, treatments in field_treatment_plot_sensor_map.items():
        for treatment, plots in treatments.items():
            dataset_id = f"{field}_trt{treatment}"
            ensure_dataset_exists(client, dataset_id, delete_existing)
            
            for plot_number, sensor_ids in plots.items():
                table_id = f"{client.project}.{dataset_id}.plot_{plot_number}"
                schema = create_table_schema(sensor_ids)
                create_or_update_table(client, table_id, schema, delete_existing)

def main():
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        yaml_path = os.path.join(script_dir, '..', '..', 'sensor_mapping.yaml')
        sensor_mapping = load_sensor_mapping(yaml_path)
        client = create_bigquery_client()

        delete_flag = input("Do you want to delete existing tables/datasets before creating new ones? (y/n): ").lower() == 'y'

        if delete_flag:
            confirmation1 = input("Are you sure you want to delete existing tables/datasets? (y/n): ").lower()
            if confirmation1 == 'y':
                confirmation2 = input("This action is irreversible. Type 'y' again to confirm deletion: ").lower()
                if confirmation2 == 'y':
                    logger.info("Deletion confirmed. Proceeding with table creation (including deletion of existing ones).")
                    process_sensor_mapping(client, sensor_mapping, delete_existing=True)
                else:
                    logger.info("Deletion cancelled. Proceeding with table creation without deleting existing ones.")
                    process_sensor_mapping(client, sensor_mapping, delete_existing=False)
            else:
                logger.info("Deletion cancelled. Proceeding with table creation without deleting existing ones.")
                process_sensor_mapping(client, sensor_mapping, delete_existing=False)
        else:
            logger.info("Proceeding with table creation without deleting existing ones.")
            process_sensor_mapping(client, sensor_mapping, delete_existing=False)

        logger.info("Process completed successfully")
    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()