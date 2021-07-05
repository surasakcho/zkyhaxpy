from zkyhaxpy import dttm_tools
from zkyhaxpy.io_tools import create_folders
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

    assert(type(log_sep)==str)
    
    create_folders(log_path)
    
    #create header of log file if not existing
    if os.path.exists(log_path) == False:
        print(f'{log_path} is not found. Start logging into a new file.')
        with open(log_path, 'w') as opened_file:
            opened_file.write(log_sep.join(['timestamp', 'log_level'] + log_columns) + '\n')                             
    else:
        df_log = pd.read_csv(log_path, sep=log_sep)       
        exist_log_columns = list(df_log.columns)[2:]
        assert(exist_log_columns==log_columns)
        print(f'"{log_path}" is found. Resume logging in the existing file.')
        

    #define logging behavior    
    logger = logging.getLogger(log_name)

    if log_level=='debug':
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    formatter=logging.Formatter(f'%(asctime)s{log_sep}%(levelname)s{log_sep}%(message)s')

    file_handler=logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    
    if logger.hasHandlers()==False:
        logger.addHandler(file_handler)

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
    