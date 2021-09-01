crop_snippet = [    
    '!pip install rasterio utm geopandas',
    ''
    'from zkyhaxpy import io_tools, gis_tools, colab_tools',
    'import pandas as pd',
    'import numpy as np',
    'import os',
    'import matplotlib.pyplot as plt',  
    '',
    'ln -s /home/james james'
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