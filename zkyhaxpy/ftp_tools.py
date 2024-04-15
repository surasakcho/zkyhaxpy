import ftplib
from ftplib import FTP, FTP_TLS
import sys
import os
import pandas as pd
from datetime import datetime, timedelta
from tqdm.notebook import tqdm
from zkyhaxpy.io_tools import create_folders


def ftp_connection(ftp_address, user, password, use_tls=False ):
    '''
    Create a FTP connection
    
    inputs
    ----------------------------------------
    ftp_address: 
        An FTP address without a protocal prefix.
        For example if FTP address is "ftp://abc.def.xyz", input "abc.def.xyz" as ftp_address.
    
    return
    ----------------------------------------
    ftp_connection:
        object of a ftp connection
	'''
    if use_tls == False:
        ftp_conn = FTP(ftp_address)  # connect to host, default port
    else:
        ftp_conn = FTP_TLS(ftp_address)  # connect to host, default port
    ftp_conn.login(user=user, passwd=password)# user anonymous, passwd anonymous@


    return ftp_conn



def ftp_walk(ftp_conn, dir, df_ftp_path):
    '''
    List all files (recursively) and save into an empty pandas dataframe (pandas.DataFrame()). 
    df_ftp_path is required as input because it will be used recursively.
    
    
    
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


def ftp_download(ftp_conn, ftp_path, out_path, skip_exist=True, delete_failed_file=True):
    '''
    Download file from specified ftp path.
    
    inputs
    --------------------------------------
    ftp_conn:
        an object of ftp connection
    ftp_path:
        a path of target file to be downloaded (without ftp address) 
        For example if FTP file path is "ftp://abc.def.xyz/dir/file_name.data", input "dir/file_name.data" as ftp_path.
    skip_exist:
        if True, skip downloading if output file already exists.
    delete_failed_file:
        if True, delete the output file that is unsuccesfully downloaded.
    
	'''
    if (os.path.exists(out_path)==True) & (skip_exist==True):
        print(f'{out_path} already exists.')
        return

    create_folders(out_path)
    try:
        with open(out_path, 'wb') as fp:                
            ftp_conn.retrbinary(f'RETR {ftp_path}', fp.write)
            print(f'{ftp_path} has been downloaded.')
    except Exception as e:
        if (delete_failed_file==True) & (os.path.exists(out_path)==True):
            print(e)
            os.remove(out_path)
            raise e
        
    

