import os
import shutil

def create_project_structure():
    # List of all files that should be in the repo
    repo_structure = [
        # Source files
        'crop2cloud24.src.__init__.py',
        'crop2cloud24.src.S1_data_ingestion.__init__.py',
        'crop2cloud24.src.S1_data_ingestion.ingest_data.py',
        'crop2cloud24.src.S2_stress_indices.__init__.py',
        'crop2cloud24.src.S2_stress_indices.cwsi.py',
        'crop2cloud24.src.S2_stress_indices.swsi.py',
        'crop2cloud24.src.S2_stress_indices.et.py',
        'crop2cloud24.src.S2_stress_indices.mds.py',
        'crop2cloud24.src.S3_irrigation_prediction.__init__.py',
        'crop2cloud24.src.S3_irrigation_prediction.irrigation_model.py',
        'crop2cloud24.src.models.__init__.py',
        'crop2cloud24.src.models.prediction_model.py',
        'crop2cloud24.src.utils.__init__.py',
        'crop2cloud24.src.utils.bigquery_operations.py',
        'crop2cloud24.src.utils.create_bq_tables.py',
        'crop2cloud24.src.utils.logger.py',
        
        # Root level files
        'crop2cloud24..env',
        'crop2cloud24.requirements.txt',
        'crop2cloud24.README.md',
        'crop2cloud24..gitignore',
        'crop2cloud24.pyproject.toml',
        'crop2cloud24.sensor_mapping.yaml',
        'crop2cloud24.run_pipeline.py',
        'crop2cloud24.Project_Structure.mermaid',
        'crop2cloud24.Data_Flow_Process.tsx',
        'crop2cloud24.setup.py',
        'crop2cloud24.file_structure_guide.md',
        'crop2cloud24.custom_instructions.md',
        'crop2cloud24.expand_structure.py',
    ]

    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Create a recycle bin directory
    recycle_bin = os.path.join(current_dir, "recycle_bin")
    if not os.path.exists(recycle_bin):
        os.makedirs(recycle_bin)
        print(f"Created recycle bin: {recycle_bin}")

    # Create files
    for item in repo_structure:
        full_path = os.path.join(current_dir, item)
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        # Create file if it doesn't exist
        if not os.path.exists(full_path):
            with open(full_path, 'w') as f:
                if item.endswith('__init__.py'):
                    f.write("# This file is intentionally left empty to mark this directory as a Python package.")
                else:
                    f.write(f"# This is a placeholder for {item}")
            print(f"Created file: {full_path}")
        else:
            print(f"File already exists (skipped): {full_path}")

    # Move files not in the structure to recycle bin
    for root, dirs, files in os.walk(current_dir):
        for file in files:
            full_path = os.path.join(root, file)
            relative_path = os.path.relpath(full_path, current_dir)
            if relative_path not in repo_structure and file != os.path.basename(__file__):
                recycle_path = os.path.join(recycle_bin, relative_path)
                os.makedirs(os.path.dirname(recycle_path), exist_ok=True)
                shutil.move(full_path, recycle_path)
                print(f"Moved file to recycle bin: {full_path}")

    print("Project structure update completed")

if __name__ == "__main__":
    create_project_structure()