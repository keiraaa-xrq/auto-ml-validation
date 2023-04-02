import logging
import os
from datetime import datetime


def setup_main_logger(project_name: str) -> logging.Logger:
    """
    Configure main logger for a project.
    """
    os.makedirs('./logs', exist_ok=True)
    date = datetime.today().strftime('%Y-%m-%d')
    filepath = f'./logs/{date}_{project_name}.log'
    # instantiate logger
    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)
    # define file handler and set formatter
    file_handler = logging.FileHandler(filepath)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    # add file handler to logger
    logger.addHandler(file_handler)
    return logger


def log_info(logger: logging.Logger, msg: str):
    print(msg)
    logger.info(msg)


def log_error(logger: logging.Logger, msg: str):
    print(msg)
    logger.error(msg)
