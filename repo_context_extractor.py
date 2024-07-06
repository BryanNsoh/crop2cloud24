import os
import sys

EXCLUDED_DIRS = {".git", "__pycache__", "node_modules", "venv", ".venv"}
EXCLUDED_FILES = {".gitignore", "README.md"}

def create_file_element(file_path, root_folder):
    """
    Create an XML element for a file, including its content.

    Args:
        file_path (str): The path to the file.
        root_folder (str): The root directory of the repository.

    Returns:
        str: The formatted XML element for the file.
    """
    relative_path = os.path.relpath(file_path, root_folder)
    file_name = os.path.basename(file_path)

    file_element = [
        f"    <file>\n        <name>{file_name}</name>\n        <path>{relative_path}</path>\n"
    ]

    file_element.append("        <content>\n")
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            file_element.append(file.read())
    except Exception as e:
        file_element.append(f"Error reading file: {str(e)}")
    file_element.append("\n        </content>\n")

    file_element.append("    </file>\n")
    return "".join(file_element)

def get_repo_structure(root_folder):
    """
    Get the directory structure of the repository with embedded file content.

    Args:
        root_folder (str): The root directory of the repository.

    Returns:
        str: The formatted XML structure of the repository.
    """
    structure = ["<repository_structure>\n"]

    for subdir, dirs, files in os.walk(root_folder):
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]
        level = subdir.replace(root_folder, "").count(os.sep)
        indent = " " * 4 * level
        structure.append(f'{indent}<directory name="{os.path.basename(subdir)}">\n')

        for file in files:
            if file not in EXCLUDED_FILES:
                file_path = os.path.join(subdir, file)
                file_element = create_file_element(file_path, root_folder)
                structure.append(file_element)

        structure.append(f"{indent}</directory>\n")

    structure.append("</repository_structure>\n")
    return "".join(structure)

def main():
    """
    Main function to execute the script.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    claude_project_dir = os.path.join(script_dir, "claude-project")

    if not os.path.exists(claude_project_dir):
        print(f"Error: 'claude-project' folder not found in {script_dir}")
        sys.exit(1)

    logs_folder = os.path.join(script_dir, "logs")
    if not os.path.exists(logs_folder):
        os.makedirs(logs_folder)

    output_file = os.path.join(logs_folder, "repository_context.txt")

    repo_structure = get_repo_structure(claude_project_dir)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(repo_structure)

    print(f"Repository context has been extracted to {output_file}")

if __name__ == "__main__":
    main()