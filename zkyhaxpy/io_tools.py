#Write
import pickle
import os
import re
import shutil
import pandas as pd
from tqdm.notebook import tqdm
from collections import namedtuple


def create_folders(path):
    
    basename = os.path.basename(path)
    dirname = os.path.dirname(path)
    
    if '.' in basename:
        os.makedirs(dirname, exist_ok=True)
    else:
        os.makedirs(path, exist_ok=True)        


def get_disk_free_space_mb(path = 'c:'):
    """Return disk free space of the given path

    Returned valus is a named tuple with attributes 'total', 'used' and
    'free', which are the amount of total, used and free space, in bytes.
    """

    return shutil.disk_usage(path)[2] / (1024 * 1024)



def read_pickle(in_pickle_path):
    '''
    Read a pickle file.

    Parameters
    ----------------------------------
    in_pickle_path : path of pickle to be read

    Return
    ----------------------------------
    An object / variable of read pickle file
    '''

    pkl_file = open(in_pickle_path, 'rb')
    data = pickle.load(pkl_file)
    pkl_file.close()

    return data

def write_pickle(data, out_pickle_path, overwrite=True):
    '''
    Write a pickle file.

    Parameters
    ----------------------------------
    data : object / variable to be written
    out_pickle_path : path of pickle to be written to

    Return
    ----------------------------------
    1 : success
    '''
    if (overwrite==False) & (os.path.exists(out_pickle_path)==True):
        print(f'{out_pickle_path} is already existing. No overwriting.')
        return

    output = open(out_pickle_path, 'wb')    
    pickle.dump(data, output)
    output.close()
    print(f'{out_pickle_path} has been saved.')
    return



def list_files_re(rootpath, filename_re=None, folder_re=None ):
    '''
    rootpath : root path to lookup files
    filename_re : regular expression to search for filename
    folder_re : regular expression to search for folder

    return : a list of filepaths
    '''


    list_files = []
    for folder, _, files in os.walk(rootpath):
        for file in files:     
            if filename_re == None:
                filename_re = '.*'
            if folder_re == None:
                folder_re = '.*'
                
            if ((re.search(filename_re, file) != None) & (re.search(folder_re, folder) != None)):
                list_files.append(os.path.join(folder, file))
        
    return list_files    
        
        
def sync_to_work(src_folder, filename_re=None, work_folder=r'c:\workspace', force=False, min_work_free_space_mb=10*1024, show_exists=False, check_file=True) :
    
    dst_folder = os.path.join(work_folder, os.path.basename(src_folder))
    os.makedirs(work_folder, exist_ok=True)
    os.makedirs(dst_folder, exist_ok=True)

    list_files_src = list_files_re(src_folder, filename_re)
    print(f'Syncing {len(list_files_src)} files from "{src_folder}" -> "{dst_folder}"')
    n = 0
    for src_path in tqdm(list_files_src):
        
        
        work_disk_free_space_mb = get_disk_free_space_mb(work_folder)
        src_file_size_mb = os.path.getsize(src_path) / (1024 * 1024)
        assert(work_disk_free_space_mb > min_work_free_space_mb)
        assert(src_file_size_mb < work_disk_free_space_mb)


        dst_path = os.path.join(dst_folder, os.path.basename(src_path))
        if os.path.exists(dst_path)==True and force==False:
            if show_exists==True:
                print(f'{dst_path} already exists.')
        else:
            try:                
                shutil.copy2(src_path, dst_path)
                print(f'{dst_path} has been copied.')
                n = n + 1
            except Exception as e:
                print(f'Error : cannot copy {src_path}!')
                print(e)
    
    if check_file==True:
        if filename_re == None :
            file_type = 'parquet'        
            list_error_files = check_files(dst_folder, file_type)
            assert(len(list_error_files)==0)

    print(f'{n} files have been synced "{src_folder}" -> "{dst_folder}" completely.')
    return dst_folder



def check_files(folder, file_type='parquet', same_columns=True):
    list_error_files = []
    list_checked_files = []
    if (file_type == 'parquet') | (file_type.lower() == '.parquet'):
        list_files = list_files_re(folder, '.parquet')
        if same_columns == True:
            df_temp = pd.read_parquet(list_files[0])
            init_col = list(df_temp.columns)[0]
            del(df_temp)
            for filepath in tqdm(list_files):
                try:
                    pd.read_parquet(filepath, columns=[init_col])
                    list_checked_files.append(filepath)
                except:
                    list_error_files.append(filepath)
                    print(f'Cannot read file {filepath}!')
        else:
            print('This check_files() version supports only same_columns = True!')
            assert(False)
    else:
        print('This check_files() version supports only file_type = parquet!')
        assert(False)
    if len(list_checked_files) == 0:
        print('No file has been checked! Please recheck folder and file_type.')
        assert(False)

    return list_error_files