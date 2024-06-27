- See the Data_Flow_Process.tsc to understand the full structure of the project repo
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