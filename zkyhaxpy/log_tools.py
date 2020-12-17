from pierpy import dttm_tools
from pierpy.io_tools import create_folders
# from pierpy import dttm_tools
# from io_tools import create_folders
import os
import logging
import numpy as np
import pandas as pd

def get_logger(log_name, log_path, log_columns, log_level='debug', log_sep='|'):
    '''
    Create a logger using file_handler.
    '''

    create_folders(log_path)
    logger = logging.getLogger(log_name)

    if log_level=='debug':
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    formatter=logging.Formatter('%(asctime)s|%(levelname)s|%(message)s')

    file_handler=logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    try:
        pd.read_csv(log_path, sep=log_sep)
    except Exception as e:
        print(f'{log_path} is not found. Start logging into a new file.')
        print(e)
        logger.info(log_formatter(dict(zip(log_columns, log_columns)), log_columns))
        
    return logger


def log_formatter(dict_log_messages, log_columns, sep='|'):
    '''
    Formatter of a dict messages to add into log as columns. Columns are separated by '|' (pipe).
    '''
    list_log_messages = [dict_log_messages.get(col, '') for col in log_columns]
    assert(len(list_log_messages) == len(log_columns))

    return sep.join(np.array(list_log_messages).astype(str))

def log_shutdown():
    logging.shutdown()
    