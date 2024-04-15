crop_snippet = [        
    '## for all ##',    
    "path_config = PATH",
    'from configparser import SafeConfigParser',
    'parser = SafeConfigParser()',
    'parser.read(path_config)',    
    '',
    'from zkyhaxpy import io_tools, pd_tools, np_tools, console_tools, timer_tools, json_tools, dict_tools, log_tools',        
    'import pandas as pd',
    'import numpy as np',
    'from tqdm.notebook import tqdm',
    'import os',
    'import shutil',
    'import re',
    'import matplotlib.pyplot as plt',  
    'import seaborn as sns',
    'from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union, overload',
    '',
    '## for gis tasks##',
    'from zkyhaxpy import gis_tools',
    'import geopandas as gpd',
    'import rasterio',    
    '',
    '## for colab ##',
    'from zkyhaxpy import colab_tools',
    'colab_tools.mount_drive()',
    
    '## if new envi, install these libs ##',
    '#!pip install rasterio utm geopandas piercrop',    
    '',
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
