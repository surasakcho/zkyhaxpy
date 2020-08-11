import os
from google.colab import auth
from google.colab import drive
from zkyhaxpy.io_tools import list_files_re



def check_dup_files(folder, file_nm_prefix=None, file_nm_extension=None, remove=False):
    if file_nm_prefix==None:
        file_nm_prefix = '.*'
    if file_nm_extension==None:
        file_nm_extension = '.*'
        
    list_file_paths = list_files_re(folder, file_nm_prefix + '.*\([0-9]{1,}\)\.' + file_nm_extension)    
    
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
