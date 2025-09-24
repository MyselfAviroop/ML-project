import logging
import os
from datetime import datetime

# Logs folder path
logs_path = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_path, exist_ok=True)

# Log file path
LOG_FILE = f"{datetime.now().strftime('%m%d%Y__%H%M%S')}.log"
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s]  %(lineno)d - %(levelname)s - %(message)s",
    level=logging.INFO,
)
