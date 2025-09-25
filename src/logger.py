import logging
import os
from datetime import datetime

# Logs folder path
logs_path = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_path, exist_ok=True)

# Log file path with timestamp
LOG_FILE = f"{datetime.now().strftime('%m%d%Y__%H%M%S')}.log"
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# Create handlers: file + console
file_handler = logging.FileHandler(LOG_FILE_PATH)
console_handler = logging.StreamHandler()

# Common log format
formatter = logging.Formatter("[%(asctime)s] %(name)s:%(lineno)d - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Root logger setup
logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])

# Module-level logger
logger = logging.getLogger(__name__)
