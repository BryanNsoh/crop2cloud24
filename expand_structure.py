import os
import shutil

def expand_structure():
    # Get the current directory (claude-project) and its parent
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)

    # The main project directory is now the parent of claude-project
    crop2cloud_dir = parent_dir

    # List of files to exclude from expansion
    exclude_files = [
        'expand_structure.py',
        'create_project_structure.py',
        'custom_instructions.md',
        'file_structure_guide.md',
        'Project_Structure.mermaid',
        'Data_Flow_Process.tsx'
    ]

    # Preserve .venv directory if it exists
    venv_dir = os.path.join(crop2cloud_dir, '.venv')
    if os.path.exists(venv_dir):
        temp_venv_dir = os.path.join(parent_dir, '.temp_venv')
        shutil.move(venv_dir, temp_venv_dir)

    # Process each file in the current directory (claude-project)
    for filename in os.listdir(current_dir):
        if filename in exclude_files:
            continue

        current_path = os.path.join(current_dir, filename)
        if os.path.isfile(current_path):
            # Remove the 'crop2cloud24.' prefix if it exists
            if filename.startswith('crop2cloud24.'):
                filename = filename[12:]

            # Handle special cases for .env and .venv
            if filename in ['.env', '.venv']:
                new_filename = filename
                new_dir = crop2cloud_dir
            else:
                # Split the filename, preserving the extension
                parts = filename.rsplit('.', 1)
                if len(parts) > 1:
                    name, extension = parts
                    parts = name.split('.')
                    new_filename = f"{parts[-1]}.{extension}"
                    new_dir = os.path.join(crop2cloud_dir, *parts[:-1])
                else:
                    new_filename = filename
                    new_dir = crop2cloud_dir

            # Create the new directory if it doesn't exist
            os.makedirs(new_dir, exist_ok=True)
            
            # The new full path for the file
            new_path = os.path.join(new_dir, new_filename)
            
            # Copy the file
            shutil.copy2(current_path, new_path)
            print(f"Copied '{filename}' to '{os.path.relpath(new_path, parent_dir)}'")

    # Restore .venv directory if it was preserved
    if os.path.exists(temp_venv_dir):
        shutil.move(temp_venv_dir, venv_dir)

    print("Structure expansion completed")

if __name__ == "__main__":
    expand_structure()