import os
import shutil
import tempfile

def collapse_structure():
    # Get the current directory (where the script is run from)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    claude_project_dir = os.path.join(current_dir, "claude-project")

    # Create a temporary directory for backup
    with tempfile.TemporaryDirectory() as temp_dir:
        # If claude-project exists, move it to the temp directory
        if os.path.exists(claude_project_dir):
            temp_claude_project = os.path.join(temp_dir, "claude-project-backup")
            shutil.move(claude_project_dir, temp_claude_project)
            print(f"Existing claude-project backed up to temporary directory")

        # Create a fresh claude-project directory
        os.makedirs(claude_project_dir)
        print(f"Created fresh claude-project directory")

        # List of directories and files to exclude
        exclude_list = ['.git', '.venv', '__pycache__', 'claude-project', 'logs']
        exclude_extensions = ['.pyc', '.log']  
        # Walk through the current directory structure
        for root, dirs, files in os.walk(current_dir):
            # Remove excluded directories
            dirs[:] = [d for d in dirs if d not in exclude_list]

            for file in files:
                # Skip excluded files
                if any(file.endswith(ext) for ext in exclude_extensions) or any(excl in root.split(os.sep) for excl in exclude_list):
                    continue

                # Get the full path of the file
                full_path = os.path.join(root, file)
                
                # Get the relative path from the current directory
                rel_path = os.path.relpath(full_path, current_dir)
                
                # Create the flattened file name
                flattened_name = "crop2cloud24." + rel_path.replace(os.path.sep, ".")
                
                # Create the new path in claude-project
                new_path = os.path.join(claude_project_dir, flattened_name)
                
                # Ensure the directory exists
                os.makedirs(os.path.dirname(new_path), exist_ok=True)
                
                # Copy the file
                shutil.copy2(full_path, new_path)
                print(f"Copied '{rel_path}' to '{flattened_name}'")

        # Copy collapse_structure.py to the claude-project directory
        collapse_structure_path = os.path.join(claude_project_dir, "crop2cloud24.collapse_structure.py")
        shutil.copy2(__file__, collapse_structure_path)
        print(f"Copied 'collapse_structure.py' to 'crop2cloud24.collapse_structure.py'")

        print("Structure collapse completed")

        # Here, the temporary directory (including the old claude-project backup) will be automatically deleted

if __name__ == "__main__":
    collapse_structure()