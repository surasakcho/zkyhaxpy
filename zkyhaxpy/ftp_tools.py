import ftplib
from ftplib import FTP
import sys
import pandas as pd
from datetime import datetime, timedelta



def ftp_walk(ftp, dir, df):
    '''
    List all files (recursively) and save into df dataframe. df is required to define as global object.
    '''
    dirs = []
    nondirs = []
    for item in ftp.mlsd(dir):
        if item[1]['type'] == 'dir':
            dirs.append(item[0])            
        else:
            nondirs.append(item[0])
            id = len(df)
            df.at[id, 'dir'] = dir
            df.at[id, 'filename'] = item[0]
            str_mod_date = str(item[1]['modify'])[:14]          
            #df.at[id, 'mod_date'] = datetime.strptime(str_mod_date, '%Y%m%d')
            df.at[id, 'mod_date'] = datetime.strptime(str_mod_date, '%Y%m%d%H%M%S') + timedelta(hours=7)
    if nondirs:        
        print('{} : {} files.'.format(dir,len(nondirs)))
        #print('\n'.join(sorted(nondirs)))
        
    else:
        # print(dir, 'is empty')
        pass
    for subdir in sorted(dirs):
        ftp_walk(ftp, '{}/{}'.format(dir, subdir), df)


def main() :
    global df 
    df = pd.DataFrame(columns = ['dir', 'filename', 'mod_date'])

    ftp = FTP('202.29.107.23')  # connect to host, default port
    #ftp.connect(port=21)       
    ftp.login(user='pier', passwd='gistda')# user anonymous, passwd anonymous@

    ftp_walk(ftp, '/', df)