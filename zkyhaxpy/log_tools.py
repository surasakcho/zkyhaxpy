from pierpy import dttm_tools
from pierpy.io_tools import create_folders
# from pierpy import dttm_tools
# from io_tools import create_folders
import os
import logging

def get_logger(log_name, log_filepath, log_level='debug'):

    create_folders(log_filepath)
    logger = logging.getLogger(log_name)

    if log_level=='debug':
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    formatter=logging.Formatter('%(asctime)s|%(levelname)s|%(message)s')

    file_handler=logging.FileHandler(log_filepath)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    return logger


def shutdown():
    logging.shutdown()
    