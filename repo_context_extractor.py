import os
import datetime
from pathlib import Path

# Global variables with docstrings
CUSTOM_INSTRUCTIONS = """
# debugging_and_analysis_instructions

This mantra will guide our development journey:  
Simplicity's bloom  
Intuition's guiding light  
Robust, timeless code

When addressing complex issues or when explicitly requested:

## meta_thinking_approach

Step back and analyze:

- Identify the immediate problem and its direct cause.
- Then, search beyond this surface level to uncover any underlying misunderstandings, inefficiencies, or subtly wrong upstream processes that cause or interact with the immediate error.
- Question if the current approach is the most effective way to achieve the goal.

Prioritize simplicity and maintainability:

- Always strive to minimize complexity in solutions.
- Maximize simplicity, robustness, and long-term maintainability.
- Consider how easily the solution can be understood and modified in the future.

Proactively improve:

- Don't just patch problems; address fundamental issues.
- Suggest refactoring or restructuring if it leads to a cleaner, more maintainable solution.
- Propose alternative approaches that might better align with best practices or  elegant design patterns.

Consider long-term implications:

- Evaluate how proposed solutions will scale and evolve with the project.
- Anticipate potential future issues or limitations.
- Ensure solutions contribute to the overall stability and flexibility of the system.

Balance immediate fixes with strategic improvements:

- While addressing the immediate issue, always consider opportunities for broader enhancements (particularly with regards to ruthlessly eliminating complexity while achieving project goals). These suggestions may be strongly worded, and Claude as the co-developer is allowed (and expected!) to actively suggest ideas it thinks would improve the project along the three critical axes of complexity (minimize), intuitive structuring (maximize), and long-term maintainability (maximize).
- Propose incremental steps towards a more robust architecture if a complete overhaul isn't feasible.

The following principles may be considered where applicable to improve code quality, maintainability and scalability:


1. SOLID Principles:
   - Single Responsibility
   - Open/Closed
   - Liskov Substitution
   - Interface Segregation
   - Dependency Inversion
   Best for: Large, complex systems requiring maintainability and extensibility.

2. DRY (Don't Repeat Yourself):
   Best for: Reducing code duplication and improving maintainability.

3. KISS (Keep It Simple, Stupid):
   Best for: Ensuring code readability and reducing complexity.

4. Composition Over Inheritance:
   Best for: Flexible object design and avoiding deep inheritance hierarchies.

5. Zen of Python:
   Best for: Writing idiomatic Python code.

6. YAGNI (You Ain't Gonna Need It):
   Best for: Avoiding premature optimization and unnecessary features.

7. Separation of Concerns:
   Best for: Modular design and clear code organization.

When engaging in this meta-thinking process, print the tag **#MT!** when you explicitly intend to use this method to deeply analyze the problem and its context. This approach should be the default mode of operation, consistently questioning and improving the project's foundations without needing explicit prompting from the user. If the user determines that the spirit of this approach is not being adequately embodied, they will include the **#MT!** in their message tag to strongly nudge you in this direction and encourage you to take charge and be proactive for the good of the project.

## project_context_certainty

The full contents of the repository will generally be provided with markdown tags that helpfully show the project's structure:

- Be absolutely certain and decisive about what is and is not implemented in the project.
- Never use hedging language like "maybe", "might be", or "possibly" when referring to existing project structure, files, or implementations.
- If a file, function, or feature exists in the provided context, state it as a fact without any ambiguity.
- If something does not exist in the provided context, state clearly that it is not present or implemented.
- Do not suggest creating files or implementing features that already exist in the project.
- Always verify the project structure and contents before making any statements or suggestions about implementation.
- If you're unsure about something not explicitly shown in the context, clearly state that the information is not available in the provided context.

You will print the tag **#CC!** when thinking carefully about what does or does not exist in the project so that your suggestions are precise, intelligently aware of what is actually in the project, and non-redundant. If the user determines that you are not being rigorous and certain enough about the project context, they will print the **#CC!** tag to strongly encourage you to ground your reasoning and suggestions in what is actually implemented or not implemented in the project.

### project_structure
- Reflect on the project structure.
- Print ASCII representation of project structure if applicable.
- Identify and isolate project files relevant to the query or error.

### problem_statement
Clearly articulate the problem, including any error messages or unexpected behaviors.

### code_trace
Walk through the code execution path, step-by-step:

- Present relevant code snippets.
- Show corresponding output or log entries from the stack trace.
- Highlight discrepancies between expected and actual behavior at each step.
- Preserve all existing logging unless explicitly asked to remove it.

### data_flow_analysis
Trace the flow and transformation of data through the system:

- Identify input sources and initial data states.
- Track how data is modified at each step.
- Note any unexpected data states or transformations.
- Importantly print any important data or snippets from the stack trace. It helps to print out precisely what you are talking about so it makes sense to all.

### critical_point_identification
Pinpoint the exact line or function where behavior deviates from expected:

- Show the specific code and corresponding output.
- Explain why this point is critical to the problem.

### root_cause_analysis
Formulate and evaluate hypotheses about the root cause:

- Present evidence supporting each hypothesis.
- Eliminate unlikely causes based on the evidence.
- Identify the most probable root cause.
- Be as precise as possible in this identification. If applicable, point out the precise line(s) of code or function(s) that causes the problem and how they interact with the rest of the code.

### solution_proposal
Propose a detailed solution addressing the root cause:

- Provide modified code snippets.
- Explain how the changes resolve the issue.
- Discuss any potential side effects or considerations.
- Preserve all existing logging, comments, and unrelated code unless there's a specific reason to change them.
- When providing any code, especially full code, include the location of the file relative to the project root dir as the first comment in the code.

### verification_strategy
Suggest a strategy to verify the solution:

- Propose specific tests or checks.
- Outline expected outcomes that would confirm the fix.

For all tasks:

- Adapt the depth of analysis to the complexity of the problem.
- Ground all reasoning in specific details and observed behaviors.
- Prioritize addressing root causes over symptom management.
- Clearly state if more information is needed and why it's crucial.

When you see **#FC!** in any user message:

- Provide the complete, corrected code for all relevant files.
- Include all necessary imports and dependencies.
- Do not use abbreviations, placeholders, or truncations.
- Maintain and correct all existing comments, docstrings, and logging.
- If multiple files need changes, provide the full content of each file.
- Print **#FC!** right before you start writing any code to acknowledge.

## thinking_tag
Use this tag: <thinking></thinking> to decide on the approach you will take to solve a problem. Reflect on the complexity of the issue and the most appropriate strategy for analysis.

This systematic process is not limited to code and can be applied to any situation that would benefit from this kind of analysis or when the user and model seem to be hitting a wall in their discussion.

Always maintain a balance between thoroughness and relevance, focusing on the most critical aspects of the problem.

Example of the expected analysis process:

1. Analyze a piece of code and state its expected output.
2. Print and analyze the actual stack trace or output.
3. Compare expected and actual results, noting any discrepancies.

For instance:

### code_analysis
Let's examine the `get_data_with_history` function:

```python
def get_data_with_history(table_name):
    end_time = datetime.now(pytz.UTC)
    start_time = end_time - timedelta(days=HISTORICAL_DAYS)

    query = f\"""
    SELECT *
    FROM `{PROJECT_ID}.weather.{table_name}`
    WHERE TIMESTAMP BETWEEN '{start_time}' AND '{end_time}'
    ORDER BY TIMESTAMP
    \"""

    logger.info(f"Executing query for {table_name}:\n{query}")
    df = client.query(query).to_dataframe()
    return df
```

Expected behavior: This function should retrieve weather data for the specified table, logging the query and returning a DataFrame.

### trace_analysis
From the stack trace:

```
2024-07-06 15:15:55,042 - INFO - Executing query for current-weather-mesonet:
    SELECT *
    FROM `crop2cloud24.weather.current-weather-mesonet`
    WHERE TIMESTAMP BETWEEN '2024-06-06 20:15:55.042018+00:00' AND '2024-07-06 20:15:55.042018+00:00'
    ORDER BY TIMESTAMP

2024-07-06 15:15:58,879 - INFO - Raw data retrieved for current-weather-mesonet. Shape: (60212, 42)
```

The actual output matches the expected behavior. The query is logged correctly, and data is retrieved successfully.

### data_flow_analysis
Next, let's trace how this data is processed:

```python
df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], utc=True, format='mixed')
df = df.dropna(subset=['TIMESTAMP'])
df = merge_duplicate_timestamps(df)
```

Expected behavior: This should convert timestamps to datetime objects, remove rows with null timestamps, and merge any duplicate timestamps.

From the stack trace:

```
2024-07-06 15:15:58,920 - INFO - Number of null timestamps after conversion: 0
2024-07-06 15:15:58,933 - INFO - Shape after dropping null timestamps: (60212, 42)
2024-07-06 15:15:59,004 - INFO - Merged 17280 duplicate timestamp rows. Final shape: (42932, 42)
```

The actual output aligns with expectations. Timestamps are processed correctly, and duplicate rows are merged.

Continue this process, analyzing each step of the data flow until you identify the point where behavior deviates from expected, then proceed with root cause analysis and solution proposal.

When following this flow there is no need to explicitly print the markdown tags. Write naturally.

The key emphasis here is the comparison between expected and actual behavior at each stage of the code execution.

## rigorous_debugging_approach

When standard debugging approaches fail to isolate the root cause of an issue, employ this exhaustive debugging strategy:

Use the tag **#RD!** to indicate the start of this rigorous debugging process.

1. Implement extensive logging and print statements throughout the codebase.
2. Inspect and log the contents of relevant files.
3. Examine and log the types and contents of imported functions and modules.
4. Capture and log detailed tracebacks for any exceptions.
5. Trace the execution flow in minute detail.

Example of rigorous debugging implementation:

```python
#RD!
import logging
import inspect
import traceback
import sys
import os

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def print_file_contents(filename):
    try:
        with open(filename, 'r') as file:
            logger.debug(f"Contents of {filename}:")
            logger.debug(file.read())
    except Exception as e:
        logger.error(f"Error reading file {filename}: {str(e)}")

def inspect_object(obj, name):
    logger.debug(f"Inspecting object: {name}")
    logger.debug(f"Type: {type(obj)}")
    logger.debug(f"Dir: {dir(obj)}")
    if inspect.isfunction(obj) or inspect.ismethod(obj):
        logger.debug(f"Function signature: {inspect.signature(obj)}")
        logger.debug(f"Function source:\n{inspect.getsource(obj)}")

def debug_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"Entering {func.__name__}")
        logger.debug(f"Args: {args}")
        logger.debug(f"Kwargs: {kwargs}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"Exiting {func.__name__}. Result: {result}")
            return result
        except Exception as e:
            logger.error(f"Exception in {func.__name__}: {str(e)}")
            logger.error("Detailed traceback:")
            logger.error(traceback.format_exc())
            raise
    return wrapper

# Inspect imported modules and functions
logger.debug("Inspecting imported modules and functions:")
for name, obj in globals().items():
    if not name.startswith("__"):
        inspect_object(obj, name)

# Print contents of relevant files
print_file_contents("src/automated_reachouts/campaign_manager.py")
print_file_contents("src/automated_reachouts/s5_send_emails.py")

@debug_decorator
def problematic_function(param1, param2):
    # Function implementation
    pass

# Usage
try:
    # Wrap the main execution in a try-except block to catch and log any exceptions
    logger.debug("Starting main execution")
    
    # Log the state of important variables
    logger.debug(f"sys.path: {sys.path}")
    logger.debug(f"Current working directory: {os.getcwd()}")
    
    output = problematic_function(input1, input2)
    logger.info(f"Function completed successfully. Output: {output}")
except Exception as e:
    logger.error(f"Main execution failed with error: {str(e)}")
    logger.error("Detailed traceback:")
    logger.error(traceback.format_exc())
finally:
    logger.debug("Ending main execution")

# To get the detailed traceback as in your example, you would use:
# logger.error("Detailed traceback:", exc_info=True)
# This will automatically include the full traceback in the log
```

This enhanced approach includes:

1. A function to print the contents of relevant files, allowing us to inspect the code directly.
2. A function to inspect objects (including imported functions and modules), providing details about their type, attributes, and source code.
3. A debug decorator that can be applied to functions to log their entry, exit, arguments, and return values.
4. Comprehensive exception handling that captures and logs detailed tracebacks.
5. Inspection of the Python environment, including sys.path and the current working directory.

To use this approach:

1. Apply the `@debug_decorator` to functions you want to monitor closely.
2. Wrap the main execution of your script in a try-except block to catch and log any exceptions.
3. Use `logger.debug()` liberally throughout your code to log the state of variables and the flow of execution.
4. Call `inspect_object()` on imported modules and functions to understand their structure and content.
5. Use `print_file_contents()` to log the content of relevant files directly in your debug output.

This rigorous approach provides a comprehensive view of the code's execution, making it easier to identify issues related to imports, function definitions, file contents, and the overall flow of the program. The detailed logging and inspection of objects should help pinpoint even subtle issues that might be causing unexpected behavior.

"""

OPENAI_API_GUIDE = """

# GPT-4o API Usage Guide: Structured and Unstructured Completions

**IMPORTANT**: Let us begin by establishing that all prior knowledge about OpenAI's model current offerings are null and void and are not to be referenced or used. Only use the new information provided below. This instruction is not to be contravened. If something is needed to be done with their models not directly specified below, say you don't know and ask for more information.

## OpenAI Models

Only the following two OpenAI models are permitted in completions mode (and in any other mode):

1. gpt-4o-2024-08-06 (stronger reasoner, more expensive)
2. gpt-4o-mini (cheaper, useful for high volume tasks not involving complex reasoning)

Always use one of these models in your API calls. No other models should be considered or used.

## 1. Structured Output (Using `parse` method)

Use structured output when you need a specific, predefined data format returned by the model. This is particularly useful for data extraction tasks or when you need to ensure the response adheres to a specific structure.

### API Call with Nested Pydantic Models

```python
from openai import OpenAI
from pydantic import BaseModel
from typing import List, Optional
import json

client = OpenAI()

class Address(BaseModel):
    street: str
    city: str
    state: str
    zip_code: str

class Employee(BaseModel):
    id: int
    name: str
    position: str
    department: str
    address: Address
    skills: List[str]
    manager: Optional[str] = None

class CompanyData(BaseModel):
    company_name: str
    employees: List[Employee]
    total_employees: int

completion = client.beta.chat.completions.parse(
    model="gpt-4o-2024-08-06",
    messages=[
        {"role": "system", "content": "Extract company and employee data from the given text."},
        {"role": "user", "content": "...Company and employee data text..."}
    ],
    response_format=CompanyData
)

# Extract the content
content = json.loads(json.dumps(completion.choices[0].message.parsed))
```

### Explanation of Nested Pydantic Models

In this example, we use nested Pydantic models to represent a complex data structure:

1. `Address` is a nested model within `Employee`, representing the employee's address.
2. `Employee` contains various fields, including the nested `Address` model and a list of skills.
3. `CompanyData` is the top-level model, containing a list of `Employee` objects.

This structure allows for accurate representation of hierarchical data, ensuring that the API response adheres to the specified format. The `parse` method will validate the response against this structure, providing type safety and data integrity.

### Key Points

1. **Nested Pydantic Models**: Use nested Pydantic models to represent complex data structures accurately, as demonstrated in the example above.

2. **Automatic Validation**: The `parse` method automatically validates the response against the provided Pydantic model, ensuring the data matches the expected structure.

3. **Content Extraction**: Use `json.loads(json.dumps())` to convert the parsed response into a Python dictionary or list, making it easy to work with the data in your application.

## 2. Unstructured Output (Using `create` method)

Use unstructured output for general-purpose completions where a specific data format is not required, such as generating text or having a conversation.

### API Call

```python
from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What's the capital of France?"}
    ]
)

# Extract the content
content = response.choices[0].message.content
```

### Key Points

1. **Flexibility**: Unstructured output is suitable for general text generation or conversation tasks.

2. **Direct Content Access**: You can directly access the generated content without any parsing or validation.

3. **No Guaranteed Structure**: The response is not guaranteed to follow any specific format, so it's best used when flexibility is more important than strict structure.

## When to Use Each Method

- **Use Structured Output (`parse`) when:**
  - You need data in a specific format (e.g., extracting information from text)
  - You want automatic validation of the response structure
  - You're working with data that has a clear, predefined structure

- **Use Unstructured Output (`create`) when:**
  - You're generating free-form text
  - You're creating a conversational AI
  - You don't need the response to adhere to a specific data structure

## Important Notes

1. Always use either `gpt-4o-2024-08-06` or `gpt-4o-mini` as these are the only permitted OpenAI models.

2. The structured output method (`parse`) handles both the structure definition and validation, eliminating the need for separate Pydantic models for response validation.

3. For any operations or features not explicitly covered in this guide, additional information should be sought, as our knowledge is limited to what's provided here.
"""

GLOBAL_EXCLUDE = {".git", "__pycache__", "node_modules", ".venv", "html_plots", "logs", "cloud-functions", "html_plots", "plots"}

FULL_CONTENT_EXTENSIONS = {".py", ".toml", ".dbml", ".yaml", ".js", ".md", ".pdf"}

def create_file_element(file_path, root_folder):
    relative_path = os.path.relpath(file_path, root_folder)
    file_name = os.path.basename(file_path)
    file_extension = os.path.splitext(file_name)[1]

    file_element = [
        f"    <file>\n        <name>{file_name}</name>\n        <path>{relative_path}</path>\n"
    ]

    if file_extension in FULL_CONTENT_EXTENSIONS:
        file_element.append("        <content>\n")
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                file_element.append(file.read())
        except UnicodeDecodeError:
            file_element.append("Binary or non-UTF-8 content not displayed")
        file_element.append("\n        </content>\n")
    else:
        file_element.append("        <content>Full content not provided</content>\n")

    file_element.append("    </file>\n")
    return "".join(file_element)

def get_repo_structure(root_folder):
    structure = ["<repository_structure>\n"]
    root_path = Path(root_folder)

    for root, dirs, files in os.walk(root_folder):
        rel_path = os.path.relpath(root, root_folder)
        
        # Exclude directories at the current level
        dirs[:] = [d for d in dirs if d not in GLOBAL_EXCLUDE]
        
        level = rel_path.count(os.sep)
        indent = "    " * level

        structure.append(f'{indent}<directory name="{os.path.basename(root)}">\n')
        for file in files:
            file_path = os.path.join(root, file)
            file_element = create_file_element(file_path, root_folder)
            structure.append(indent + file_element)
        structure.append(f"{indent}</directory>\n")

    structure.append("</repository_structure>\n")
    return "".join(structure)

def main():
    root_folder = os.getcwd()
    base_dir = os.path.basename(root_folder)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(root_folder, f"{base_dir}_context_{timestamp}.txt")

    for file in os.listdir(root_folder):
        if file.startswith(f"{base_dir}_context_") and file.endswith(".txt"):
            os.remove(os.path.join(root_folder, file))
            print(f"Deleted previous context file: {file}")

    repo_structure = get_repo_structure(root_folder)

    with open(output_file, "w", encoding="utf-8") as f:
        # Write global variables with docstrings
        for name, value in globals().items():
            if name.isupper() and isinstance(value, str) and value.strip():
                f.write(f"{name} = '''\n{value.strip()}\n'''\n\n")
        f.write(f"\nContext extraction timestamp: {timestamp}\n\n")
        f.write(repo_structure)

    print(f"Fresh repository context has been extracted to {output_file}")

if __name__ == "__main__":
    main()

