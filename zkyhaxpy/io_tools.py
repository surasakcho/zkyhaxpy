#Write
import pickle
import os
import re
import shutil
import pandas as pd
import random
from tqdm.notebook import tqdm
from collections import namedtuple
from zipfile import ZipFile 

from distutils.dir_util import copy_tree
from warnings import warn




def create_folders(*args):
    '''
    Create folders for given path(s). 
    '''
    for path in args:
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



         
         
         
def get_list_files(rootpath, filename_re=None, folder_re=None, return_df=False ):
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
    
    print(f'Total of {len(list_files)} files have been listed.')
    if return_df==False:
        return list_files
    else:
        return filepaths_to_df(list_files)
         


def get_list_files_re(rootpath, filename_re=None, folder_re=None, return_df=False ):
    '''
    rootpath : root path to lookup files
    filename_re : regular expression to search for filename
    folder_re : regular expression to search for folder

    return : a list of filepaths
    '''
    warn('get_list_files_re will be deprecated in the future.', FutureWarning, stacklevel=2)
    return get_list_files(rootpath, filename_re, folder_re, return_df)
   

def sync_folders(src_folder, dst_folder, filename_re=None, force=False, show_exists=False, random_sequence=False):        
    '''
    Sync all files in the source folder that have names matched with the given regular expression. 
    If force is true, all existing files will be overwritten.
    
    '''
    create_folders(dst_folder)

    list_files_src = get_list_files_re(src_folder, filename_re)
    
    if random_sequence==True:
        random.shuffle(list_files_src)
        
    print(f'Syncing {len(list_files_src)} files from "{src_folder}" -> "{dst_folder}"')
    n = 0
    for src_path in tqdm(list_files_src):        
        dst_path = src_path.replace(src_folder, dst_folder)
        create_folders(dst_path)
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
    
    print(f'{n} files have been synced "{src_folder}" -> "{dst_folder}" completely.')
    return dst_folder


def sync_to_work(src_folder, work_folder=r'c:\workspace', filename_re=None, force=False, min_work_free_space_mb=10*1024, show_exists=False, check_file=True) :
    
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


 
def zip_files(list_file_paths, out_zip_path): 
    '''

    Zipping all files in the given list of file paths.

    Parameters
    ----------
    list_file_paths: Array-like
        A list of file paths to be zipped.
        
    out_zip_path: str
        A string of output zip path

    Returns
    -------
    None

	'''

    # printing the list of all files to be zipped 
    if len(list_file_paths) == 0:
        print('No file path given. Exit the process.')
        return
    elif len(list_file_paths) <= 20:
        print('Following files will be zipped.')
        for filepath in list_file_paths:
            print(filepath)    
    else :
        print(f'Total of {len(list_file_paths)} files will be zipped.') 
        
  
    # writing files to a zipfile 
    with ZipFile(out_zip_path,'w') as zip: 
        # writing each file one by one 
        for filepath in list_file_paths: 
            zip.write(filepath, os.path.basename(filepath)) 
  
    print(f'All files have been zipped to {out_zip_path} successfully!')         
  
 
def zip_folder(folder, out_zip_path): 
    '''

    Zipping all files in the given folder.

    Parameters
    ----------
    folder: str
        A string of folder path
        
    out_zip_path: str
        A string of output zip path

    Returns
    -------
    None

	'''

    list_file_paths = list_files_re(folder)
    # printing the list of all files to be zipped 
    
    if len(list_file_paths) > 0:
            
        # printing the list of all files to be zipped        
        if len(list_file_paths) <= 20:
            print('Following files will be zipped.')
            for filepath in list_file_paths:
                print(filepath)    
        else :
            print(f'Total of {len(list_file_paths)} files will be zipped.') 
            
    
        # writing files to a zipfile 
        with ZipFile(out_zip_path,'w') as zip: 
            # writing each file one by one 
            for filepath in list_file_paths: 
                zip.write(filepath, filepath.replace(os.path.dirname(folder), ''))
    
        print(f'All files have been zipped to {out_zip_path} successfully!')         
    
    else:
        print(f'No file in folder {folder}')
  

def unzip(zip_path, out_folder):
    # opening the zip file in READ mode 
    with ZipFile(zip_path, 'r') as zip:    
        # extracting all the files 
        print('Extracting all the files now...')                 
        create_folders(out_folder)
        zip.extractall(path=out_folder)
        print(f'{zip_path} has been extracted to {out_folder} successfully.') 
        



def filepaths_to_df(list_files):
    '''

    Create a pandas dataframe for getting folder names, file names and file extensions of the given list file paths.

	'''
    
    df_list_files = pd.DataFrame(list_files, columns=['file_path'])
    df_list_files['file_nm'] = df_list_files['file_path'].apply(lambda path : os.path.basename(path))
    df_list_files['folder_nm'] = df_list_files['file_path'].apply(lambda path : os.path.basename(os.path.dirname(path)))
    df_list_files['file_ext'] = df_list_files['file_nm'].apply(lambda file_nm : file_nm.split('.')[-1])
    
    return df_list_files
    
    
def check_all_files_exists(*args):
    '''

    Check whether all file paths in args are already exists. If all of the paths exists, then return True. Otherwise, return False.

    Examples
    --------
    >>> check_all_files_exists(r"C:\test\20201113_1206.xlsx", r"D:\directory.txt")

	'''
    for file_path in args:
        if os.path.exists(file_path)==False:
            return False
        
    return True


def copy_if_not_exists(file_path, src_folder, if_many_source_files='error'):
    '''
    Check if a file is existing, if not, copy the file from the source folder.

    inputs
    ----------------------------------
    file_path: str or path
        a path of file to be checked
    src_folder: str or path
        a folder of source file to copy from
    if_many_source_files: str
        how to handle the process if there are many files with the same name in the source folder
        'error': will stop this process with an assertion error
        'first': will copy the first file found as the source file
        'last': will copy the last file found as the source file
    '''

    assert(if_many_source_files in ['error', 'first', 'last'])

    file_nm = os.path.basename(file_path)
    if os.path.exists(file_path):
        return
    else:
        list_src_files = get_list_files_re(src_folder, file_nm)
        
        if len(list_src_files)==0:
            print('There is no source file found. Stop copying')
            raise FileNotFoundError
        elif len(list_src_files)==1:
            shutil.copy2(list_src_files[0], file_path)                    
        else:
            if if_many_source_files=='error':
                print('There are more than one source file found. Stop copying')
                assert(len(list_src_files)==1)
            elif if_many_source_files=='first':
                shutil.copy2(list_src_files[0], file_path)                    
            elif if_many_source_files=='last':
                shutil.copy2(list_src_files[-1], file_path)                    
        
        print('File has been copied')
        
            