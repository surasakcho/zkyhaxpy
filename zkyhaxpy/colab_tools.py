import os
from google.colab import auth
from google.colab import drive
from zkyhaxpy.io_tools import get_list_files_re, filepaths_to_df
import pandas as pd
import re


def check_dup_files(folder, file_nm_prefix=None, file_nm_extension=None, remove=False):
    if file_nm_prefix==None:
        file_nm_prefix = '.*'
    if file_nm_extension==None:
        file_nm_extension = '.*'
        
    list_file_paths = get_list_files_re(folder, file_nm_prefix + '.*\([0-9]{1,}\)\.' + file_nm_extension)    
    
    if remove==True:
        print(f'Total of {len(list_file_paths)} duplicated files will be removed.')
        for filepath in list_file_paths:
            os.remove(filepath)
            print(f'{filepath} is removed.')
    else:   
        return list_file_paths
        
    
def mount_drive():
    drive.mount('/content/drive', force_remount=True)
    
def authen_gcp():
    auth.authenticate_user()




def get_list_files_re_gcs(bucket_path_prefix, storage_client, filename_re=None, folder_re=None, return_as_df=False):
    '''
    Get a list of files in Google Cloud Storage

    Parameters
    -------------------------------------
    bucket_path_prefix (str): 
        prefix of files to be listed (must start with bucket name)
        For example, if bucket name = 'test-bucket', and a file is saved at '/test-bucket/path_prefix/test1.file'
        To list all files in '/test-bucket/path_prefix/', path_prefix shall be '/test-bucket/path_prefix/'

    storage_client (google.cloud.storage.client.Client):
        Storage client that can be used to access the bucket

    filename_re (str): 
        regular expression to search for filename

    folder_re (str): 
        regular expression to search for folder


    Output
    -------------------------------------
    return : a list of tuple(filepaths
    '''

    bucket_name = bucket_path_prefix.split('/')[1]
    path_prefix = bucket_path_prefix[len(bucket_name)+2:]
    bucket = storage_client.bucket(bucket_name)

    all_blobs = list(bucket.list_blobs(prefix=path_prefix))

    list_files = []
    if len(all_blobs) == 0:
        return list_files


    df_files = pd.DataFrame(all_blobs, columns=['blob'])
    df_files['path'] = df_files['blob'].astype(str).str.split(',', expand=True).loc[:, 1].str.strip()  

    if filename_re == None:
        filename_re = '.*'
    if folder_re == None:
        folder_re = '.*'
    for filepath in list(df_files['path']):   
        file_nm = os.path.basename(filepath)
        folder = os.path.dirname(filepath)
        if ((re.search(filename_re, file_nm) != None) & (re.search(folder_re, folder) != None)):
            list_files.append(os.path.join('/'+bucket_name, folder, file_nm))

        
    if return_as_df==False:
        return list_files
    else:
        return filepaths_to_df(list_files)    
        
    