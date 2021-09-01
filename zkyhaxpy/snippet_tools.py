crop_snippet = [    
    '!pip install rasterio utm geopandas',
    ''
    'from zkyhaxpy import io_tools, pd_tools, np_tools, gis_tools, colab_tools, snippet_tools, console_tools',
    'import pandas as pd',
    'import numpy as np',
    'import os',
    'import matplotlib.pyplot as plt',  
    '',
    '!ln -s /content/drive/MyDrive/!Surasak-PIER/Crop-Insurance/NDA-Data/vew_plant_info_official_polygon_disaster_all_rice_by_year /plant_info'
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