from setuptools import setup, find_packages

# Read the contents of your requirements file
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="crop2cloud24",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "run_pipeline=src.run_pipeline:main",
        ],
    },
    # Include additional files in the package
    package_data={
        "": ["*.yaml", "*.md", "*.tsx", "*.mermaid"],
    },
    # Specify files to include in the distribution
    data_files=[
        ('', ['README.md', 'requirements.txt', '.gitignore', 'pyproject.toml', 
              'sensor_mapping.yaml', 'Project_Structure.mermaid', 'Data_Flow_Process.tsx',
              'file_structure_guide.md', 'custom_instructions.md']),
    ],
)