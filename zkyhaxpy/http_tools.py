import requests
import os

from zkyhaxpy import io_tools, gis_tools


def download_from_url(url, dest_path, skip_exists=True):
    '''
    Download a file from given url to destination path.
    '''
    
    if (os.path.exists(dest_path)==True) & (skip_exists==True):
        print(f'File {dest_path} already existed. Skip this file.')
    else:
        try:
            with requests.get(url, allow_redirects=True) as r:
                open(dest_path, 'wb').write(r.content)  
                print(f'{url} has been downloaded to {dest_path}.')
        except Exception as e:
            print(f'Cannot download {url} with error ({e}).')
    