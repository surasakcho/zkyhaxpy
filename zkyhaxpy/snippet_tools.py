crop_snippet = [    
    '## if new envi, install these libs ##',
    '#!pip install rasterio utm geopandas piercrop',    
    '',
    '## for all ##',
    'from zkyhaxpy import io_tools, pd_tools, np_tools, console_tools, timer_tools, json_tools, dict_tools',        
    'import pandas as pd',
    'import numpy as np',
    'from tqdm.notebook import tqdm',
    'import os',
    'import shutil',
    'import re',
    'import matplotlib.pyplot as plt',  
    'import seaborn as sns',
    
    '',
    '## for gis tasks##',
    'from zkyhaxpy import gis_tools',
    'import geopandas as gpd',
    'import rasterio',
    
    '',
    '## for colab ##',
    'from zkyhaxpy import colab_tools',
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
