import logging
from datetime import datetime


def setup_logger(logger: logging.Logger, filepath: str = None) -> logging.Logger:
    """
    Configure Logger.
    """
    if filepath is None:
        date = datetime.today().strftime('%Y-%m-%d')
        filepath = f'./logs/{date}.log'
    logger.setLevel(logging.INFO)
    # define file handler and set formatter
    file_handler = logging.FileHandler(filepath)
    formatter = logging.Formatter(
        '%(asctime)s : %(levelname)s : %(name)s : %(message)s')
    file_handler.setFormatter(formatter)
    # add file handler to logger
    logger.addHandler(file_handler)
    return logger


def log_info(logger: logging.Logger, msg: str):
    print(msg)
    logger.info(msg)
