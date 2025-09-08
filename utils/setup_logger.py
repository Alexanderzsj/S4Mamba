import logging
import os
import sys
import colorama
from colorama import Fore, Style
from datetime import datetime, timezone, timedelta

colorama.init()
COLORAMA_AVAILABLE = True
# try:
#     import colorama
#     from colorama import Fore, Style
#     colorama.init()
#     COLORAMA_AVAILABLE = True
# except ImportError:
#     COLORAMA_AVAILABLE = False


class ColorFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: Fore.BLUE,
        logging.INFO: 2,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
    }

    def format(self, record):
        level_color = self.COLORS.get(record.levelno, "")
        reset = Style.RESET_ALL if COLORAMA_AVAILABLE else ""
        message = super().format(record)
        return f"{level_color}{message}{reset}"


def setup_logger(name=__name__, logfile=None, level=logging.INFO, use_color=True, model_name=""):
    """
    :param name: Name of the logger (e.g., dataset name)
    :param logfile: Path to the log file, if None, no file output
    :param level: Logging level
    :param use_color: Whether to use colored console output
    :param model_name: Name of the training model to include in log messages
    :return: Logger object
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # Prevent duplicate logs

    # Clear existing handlers
    logger.handlers.clear()

    # Define log format (include model_name, exclude line number)
    log_format = "[%(levelname)s %(asctime)s Model:%(model_name)s] %(message)s"
    # Use Beijing time (CST, UTC+8)
    date_format = "%Y-%m-%d %H:%M:%S %Z"

    # Custom formatter to inject model_name
    class CustomFormatter(ColorFormatter):
        def format(self, record):
            record.model_name = model_name  # Inject model_name into log record
            return super().format(record)

    # Console output
    stream_handler = logging.StreamHandler(sys.stdout)
    if use_color and COLORAMA_AVAILABLE:
        formatter = CustomFormatter(log_format, datefmt=date_format)
    else:
        formatter = logging.Formatter(log_format, datefmt=date_format)
    # Set Beijing timezone for the formatter
    formatter.converter = lambda *args: datetime.now(timezone(timedelta(hours=8))).timetuple()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # File output (optional)
    if logfile:
        file_handler = logging.FileHandler(logfile, mode='w', encoding='utf-8')
        file_formatter = logging.Formatter(log_format, datefmt=date_format)
        file_formatter.converter = lambda *args: datetime.now(timezone(timedelta(hours=8))).timetuple()
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger