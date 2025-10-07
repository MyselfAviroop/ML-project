import sys
from src.logger import logging

# Configure logging once
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]  # logs to console
)

def error_message_detail(error: Exception, error_detail: sys = None) -> str:
    """Extract detailed error message with file and line number."""
    _, _, exc_tb = sys.exc_info() if error_detail is None else error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    return f"Error in [{file_name}] at line [{line_number}]: {error}"

class CustomException(Exception):
    def __init__(self, error: Exception, error_detail: sys = None):
        super().__init__(error)
        self.error_message = error_message_detail(error, error_detail)

    def __str__(self):
        return self.error_message
