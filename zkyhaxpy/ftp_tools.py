import ftplib
from ftplib import FTP
import sys
import os
import pandas as pd
from datetime import datetime, timedelta
from tqdm.notebook import tqdm
from zkyhaxpy.io_tools import create_folders


def ftp_connection(ftp_address, user, password):
    '''
    Create a FTP connection
	'''
    
    ftp_conn = FTP(ftp_address)  # connect to host, default port
    #ftp.connect(port=21)       
    ftp_conn.login(user=user, passwd=password)# user anonymous, passwd anonymous@


    return ftp_conn



def ftp_walk(ftp_conn, dir, df_ftp_path):
    '''
    List all files (recursively) and save into df dataframe. df is required to define as global object.
    '''
    dirs = []
    nondirs = []
    for item in ftp_conn.mlsd(dir):
        if item[1]['type'] == 'dir':
            dirs.append(item[0])            
        else:            
            if item[0] not in ('.', '..'):
                nondirs.append(item[0])
                id = len(df_ftp_path)
                df_ftp_path.at[id, 'dir'] = dir
                df_ftp_path.at[id, 'filename'] = item[0]
                str_mod_date = str(item[1]['modify'])[:14]          
                #df.at[id, 'mod_date'] = datetime.strptime(str_mod_date, '%Y%m%d')
                df_ftp_path.at[id, 'mod_date'] = datetime.strptime(str_mod_date, '%Y%m%d%H%M%S') + timedelta(hours=7)
    if nondirs:        
        print('{} : {} files.'.format(dir,len(nondirs)))
        #print('\n'.join(sorted(nondirs)))
        
    else:
        # print(dir, 'is empty')
        pass
    for subdir in sorted(dirs):
        ftp_walk(ftp_conn, '{}/{}'.format(dir, subdir), df_ftp_path)


def ftp_download(ftp_conn, ftp_path, out_path, skip_exist=True):
    '''
    Download file from specified ftp path.
	'''
    if (os.path.exists(out_path)==True) & (skip_exist==True):
        print(f'{out_path} already exists.')
        return

    create_folders(out_path)
    with open(out_path, 'wb') as fp:        
        ftp_conn.retrbinary(f'RETR {ftp_path}', fp.write)
        print(f'{ftp_path} has been downloaded.')
    

