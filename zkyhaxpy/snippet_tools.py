crop_snippet = [    
    '!pip install rasterio utm geopandas',
    ''
    'from zkyhaxpy import io_tools, pd_tools, np_tools, console_tools, timer_tools, json_tools, dict_tools, gis_tools, colab_tools',
    'import pandas as pd',
    'import numpy as np',
    'from tqdm.notebook import tqdm',
    'import os',
    'import matplotlib.pyplot as plt',  
    'import seaborn as sns',
    '',
    'colab_tools.mount_drive()',
    '!ln -s /content/drive/MyDrive/!Surasak-PIER/Crop-Insurance/NDA-Data/vew_plant_info_official_polygon_disaster_all_rice_by_year /plant_info',    
    '!ln -s /content/drive/MyDrive/!PIER /!PIER',
    '!ln -s /content/drive/MyDrive/!Surasak-PIER/Crop-Insurance /Crop-Insurance',
]

def print_snippet(module):
    '''
    Show snippet for specified module

    Available modules
    - crop    

    '''
    if module=='crop':
        for txt in crop_snippet:
            print(txt)
