import sys
from src.logger import logging

# Configure logging once
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]  # logs to console
)

def error_message_detail(error: Exception) -> str:
    _, _, exc_tb = sys.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    return f"Error in [{file_name}] at line [{exc_tb.tb_lineno}]: {error}"

class CustomException(Exception):   
    def __init__(self, error: Exception):
        super().__init__(error)
        self.error = error

    def __str__(self):
        return error_message_detail(self.error)

