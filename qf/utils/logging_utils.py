"""Utilities for logging and printing."""

import os
import sys
from datetime import datetime
from typing import TextIO


def setup_print_logging(log_dir: str = "logs") -> None:
    """Override print function to log to both console and file.

    Args:
        log_dir: Directory to store log files
    """
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Create log file with timestamp
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"print_log_{current_time}.log")

    # Open log file
    log_file_handle = open(log_file, "w", encoding="utf-8")

    # Store original print function
    original_print = print

    # Define new print function
    def custom_print(*args, **kwargs):
        # Get timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Convert args to string
        message = " ".join(str(arg) for arg in args)

        # Format log entry
        log_entry = f"{timestamp} - {message}\n"

        # Write to log file
        log_file_handle.write(log_entry)
        log_file_handle.flush()

        # Call original print
        original_print(*args, **kwargs)

    # Override print function
    sys.stdout.write = custom_print
