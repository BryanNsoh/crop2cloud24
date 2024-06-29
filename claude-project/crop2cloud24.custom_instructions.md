- This project is in the Chicago timezone but data is stored in UTC on bigquery. Thus all computations must convert from UTC to central time. This must be unambiguously understood.
- Always provide full updated code. never include placeholders in code
- See the Data_Flow_Process.tsc to understand the full structure of the project repo
- Whenever you want to do a bigquery operation think: will this operation be done repeatedly? would it do to create a general fuinction for it in bigquery operations so we can import and use?
- Always use the custom logger module logger.py for logging:
To use this logging solution in your other modules, you can simply import and use it like this:

```python
from src.utils.logger import get_logger

logger = get_logger()  # Uses the calling module's name
# or
logger = get_logger("custom_name")  # Uses a custom name

logger.info("This is an info message")
logger.error("This is an error message")
```