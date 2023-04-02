"""
Logging script for Core module
"""
import logging
import os
from datetime import datetime

from core.config import PROJECT_NAME


def _setup_console_handler(logger: logging.Logger, log_level: int) -> None:
    """
    Setup console handler
    :param logger: Logger instance
    :type logger: logging.Logger
    :param log_level: The log level
    :type log_level: int
    :return: None
    :rtype: NoneType
    """
    console_handler: logging.StreamHandler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    logger.addHandler(console_handler)


def _setup_file_handler(logger: logging.Logger, log_level: int) -> None:
    """
    Setup file handler
    :param logger: The logger instance
    :type logger: logging.Logger
    :param log_level: The log level
    :type log_level: int
    :return: None
    :rtype: NoneType
    """
    formatter: logging.Formatter = logging.Formatter(
        '[%(name)s][%(asctime)s][%(levelname)s][%(module)s][%(funcName)s][%('
        'lineno)d]: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
    current_file_directory: str = os.path.dirname(os.path.abspath(__file__))
    project_root: str = current_file_directory
    while os.path.basename(project_root) != PROJECT_NAME:
        project_root = os.path.dirname(project_root)
    current_date: str = datetime.today().strftime("%d-%b-%Y-%H-%M-%S")
    log_filename: str = f"log-{current_date}.log"
    filename_path: str = f"{project_root}/logs/{log_filename}"
    file_handler: logging.FileHandler = logging.FileHandler(filename_path)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    file_handler.flush()


def setup_logging(log_level: int = logging.DEBUG) -> None:
    """
    Setup logging
    :param log_level: Level of logging
    :type log_level: int
    :return: None
    :rtype: NoneType
    """
    logger: logging.Logger = logging.getLogger()
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(log_level)
    _setup_console_handler(logger, log_level)
    _setup_file_handler(logger, log_level)
