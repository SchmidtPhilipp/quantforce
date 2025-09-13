import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from qf.settings import DEFAULT_LOG_DIR


VERBOSITY = 2

# Default logging configurations
DEFAULT_CONSOLE_LEVELS = [
    "INFO",
    "WARNING",
    "ERROR",
    "CRITICAL",
    # "DEBUG",
]  # Show these in console by default
DEFAULT_LOG_LEVELS = [
    "DEBUG",
    "INFO",
    "WARNING",
    "ERROR",
    "CRITICAL",
]  # Log these to files by default


# Verbosity level mappings
VERBOSITY_LEVELS = {
    0: ["ERROR", "CRITICAL"],  # Only critical errors
    1: ["INFO", "WARNING", "ERROR", "CRITICAL"],  # Basic information
    2: ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],  # Detailed debugging
    3: ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],  # Maximum verbosity
}


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds color and level prefix to log messages."""

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[41m",  # Red background
        "RESET": "\033[0m",  # Reset
    }

    def format(self, record):
        # Add level prefix with color
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[levelname]}[{levelname}]{self.COLORS['RESET']}"
            )
        return super().format(record)


class FileFormatter(logging.Formatter):
    """Custom formatter for file output without color codes."""

    def format(self, record):
        # Create a copy of the record to avoid modifying the original
        record_copy = logging.LogRecord(
            name=record.name,
            level=record.levelno,
            pathname=record.pathname,
            lineno=record.lineno,
            msg=record.msg,
            args=record.args,
            exc_info=record.exc_info,
        )
        record_copy.__dict__.update(record.__dict__)

        # Add level prefix without color
        record_copy.levelname = f"[{record.levelname}]"
        return super().format(record_copy)


def setup_logging(
    verbosity: int = VERBOSITY,
    log_dir: str = DEFAULT_LOG_DIR,
    log_levels: Optional[
        List[str]
    ] = DEFAULT_LOG_LEVELS,  # Default to logging all levels
    console_levels: Optional[
        List[str]
    ] = None,  # Will be set based on verbosity if None
    email_config: Optional[Dict] = None,
    redirect_print: bool = False,
    log_format: str = "%(asctime)s - %(levelname)s - %(message)s",
    console_only: bool = False,
) -> logging.Logger:
    """
    Set up logging configuration for the entire project.

    Args:
        verbosity (int): Logging verbosity level (0-3)
            0: Only critical errors
            1: Basic information (INFO and above)
            2: Detailed debugging (DEBUG and above)
            3: Maximum verbosity (same as 2)
        log_dir (str): Directory to store log files
        log_levels (Optional[List[str]]): List of levels to log to files
        console_levels (Optional[List[str]]): List of levels to show in console
        email_config (Optional[Dict]): Configuration for email logging
        redirect_print (bool): Whether to redirect print statements to logging
        log_format (str): Format string for log messages
        console_only (bool): If True, only log to console, not to files
    """
    # Create log directory if it doesn't exist and not in console-only mode
    if not console_only:
        os.makedirs(log_dir, exist_ok=True)

    # Set console levels based on verbosity if not specified
    if console_levels is None:
        console_levels = VERBOSITY_LEVELS.get(verbosity, DEFAULT_CONSOLE_LEVELS)
    elif not isinstance(console_levels, list):
        console_levels = DEFAULT_CONSOLE_LEVELS

    # Ensure console_levels is a list for type checker
    assert isinstance(console_levels, list), "console_levels must be a list"

    # Ensure log_levels is a list for type checker
    if log_levels is None:
        log_levels = DEFAULT_LOG_LEVELS
    elif not isinstance(log_levels, list):
        log_levels = DEFAULT_LOG_LEVELS

    assert isinstance(log_levels, list), "log_levels must be a list"

    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Set root logger to capture all levels

    # Remove existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Create formatters
    console_formatter = ColoredFormatter(
        "%(asctime)s %(levelname)s %(message)s"
    )  # Simplified console format
    file_formatter = FileFormatter(log_format)  # Use custom file formatter

    # Set up console handler
    console_handler = logging.StreamHandler(sys.__stdout__)
    console_handler.setLevel(logging.DEBUG)  # Capture all levels, filter later
    console_handler.setFormatter(console_formatter)

    # Add filter for console levels
    class ConsoleLevelFilter(logging.Filter):
        def __init__(self, console_levels):
            super().__init__()
            self.console_level_nos = [
                getattr(logging, level) for level in console_levels
            ]

        def filter(self, record):
            return record.levelno in self.console_level_nos

    console_handler.addFilter(ConsoleLevelFilter(console_levels))
    logger.addHandler(console_handler)

    # Set up file handlers if not in console-only mode
    if not console_only:
        # Get current timestamp for log files
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Create a file handler for each level
        for level in log_levels:
            level_lower = level.lower()
            path = Path(log_dir) / f"{level_lower}.log"
            file_handler = logging.FileHandler(path, mode="w")
            file_handler.setLevel(getattr(logging, level))

            # Add filter for specific level
            class LevelFilter(logging.Filter):
                def __init__(self, level):
                    super().__init__()
                    self.level = level
                    self.level_no = getattr(logging, level)

                def filter(self, record):
                    # Compare the actual log level number instead of levelname
                    return record.levelno == self.level_no

            file_handler.addFilter(LevelFilter(level))

            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

        # Also create an 'all.log' that captures everything
        all_log_path = Path(log_dir) / f"all.log"
        all_log_handler = logging.FileHandler(all_log_path, mode="w")
        all_log_handler.setLevel(logging.DEBUG)
        all_log_handler.setFormatter(file_formatter)
        logger.addHandler(all_log_handler)

    # Set up email handler if configured and not in console-only mode
    if email_config and not console_only:
        from logging.handlers import SMTPHandler

        mailhost = (email_config["server"], email_config.get("port", 587))
        credentials = (email_config["email"], email_config["password"])
        secure = () if email_config.get("use_tls", True) else None

        mail_handler = SMTPHandler(
            mailhost=mailhost,
            fromaddr=email_config["email"],
            toaddrs=email_config["to"],
            subject=email_config.get("subject", "Error in Python Program"),
            credentials=credentials,
            secure=secure,
        )
        mail_handler.setLevel(logging.ERROR)
        mail_handler.setFormatter(file_formatter)
        logger.addHandler(mail_handler)

    # Redirect print statements to logging if requested
    if redirect_print:

        class LoggerWriter:
            def __init__(self, level_func, stream):
                self.level_func = level_func
                self.stream = stream

            def write(self, message):
                message = message.strip()
                if message:
                    self.level_func(message)
                    self.stream.write(message + "\n")
                    self.stream.flush()

            def flush(self):
                self.stream.flush()

        sys.stdout = LoggerWriter(logger.info, sys.__stdout__)
        sys.stderr = LoggerWriter(logger.error, sys.__stderr__)

    # Configure specific loggers for different modules
    module_loggers = {
        "qf.envs": logging.DEBUG,
        "qf.agents": logging.INFO,
        "qf.utils": logging.INFO,
        "qf.models": logging.INFO,
        "qf.trainers": logging.INFO,
    }

    for module, level in module_loggers.items():
        module_logger = logging.getLogger(module)
        module_logger.setLevel(level)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.

    Args:
        name (str): Name of the module (typically __name__)

    Returns:
        logging.Logger: Configured logger instance
    """
    return logging.getLogger(name)
