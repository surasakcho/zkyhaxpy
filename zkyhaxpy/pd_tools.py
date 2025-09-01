import pandas as pd
import numpy as np
from IPython.display import display, HTML, display_html
from tqdm.notebook import tqdm
import os
from zkyhaxpy import io_tools
import dask.dataframe as dd

def auto_adjust():
    '''
    Set column width = 100
    Max displayed rows = 100
    Max displayed columns = 100
	'''
    set_colwidth(100)
    pd.options.display.max_rows = 100
    pd.options.display.max_columns = 100



def set_max_rows(max_rows=100):
    '''
    Set max display rows
    
    Return : None
    '''

    pd.options.display.max_rows = max_rows


def set_max_columns(max_columns=100):
    '''
    Set max display columns
    
    Return : None
    '''

    pd.options.display.max_columns = max_columns



def display_html(df):
    '''
    display a dataframe as html table
    '''
    display(HTML(df.to_html()))


def inc_colwidth(inc_colwidth=20,target_colwidth=None):
    '''
    Increase column width of pandas dataframe display

    Return : None

    '''
    if target_colwidth == None:
        curr_max_colwidth = pd.get_option("display.max_colwidth") 
        new_max_colwidth = curr_max_colwidth + inc_colwidth
        pd.set_option('max_colwidth', new_max_colwidth)
    else:
        pd.set_option('max_colwidth', target_colwidth)

    print(f'Current max column width = {pd.get_option("display.max_colwidth")}')


def dec_colwidth(dec_colwidth=20,target_colwidth=None):
    '''
    Decrease column width of pandas dataframe display
    
    Return : None
    '''
    if target_colwidth == None:
        curr_max_colwidth = pd.get_option("display.max_colwidth") 
        new_max_colwidth = curr_max_colwidth - dec_colwidth
        pd.set_option('max_colwidth', new_max_colwidth)
    else:
        pd.set_option('max_colwidth', target_colwidth)   

    print(f'Current max column width = {pd.get_option("display.max_colwidth")}')


def set_colwidth(target_colwidth=100):
    '''
    Decrease column width of pandas dataframe display
    
    Return : None
    '''

    pd.set_option('max_colwidth', target_colwidth)


def get_curr_colwidth():
    '''
    Decrease column width of pandas dataframe display
    
    Return : None
    '''

    print(f'Current max column width = {pd.get_option("display.max_colwidth")}')
    
    



        
def read_parquets(file_paths=None, root_folder=None, folder_re=None, filename_re=None, columns=None, print_count=False, engine='auto', auto_dask=True, auto_dask_min_files=100, use_dask=None, progress_bar=True, sample_frac=None, random_state=88):
    '''

    Read multiple parquet files of the same template into a single pandas dataframe.
    File paths can be a list of file paths or a regular expression of file paths.
    This function is also implemented Dask's Dataframe to read files automatically when there are more than 100 files. (can be configured with parameters)

	'''
    
    if type(file_paths) == list:
        list_file_path = file_paths
    elif type(file_paths) == str:
        folder = os.path.dirname(file_paths)
        filename = os.path.basename(file_paths)
        list_file_path = io_tools.get_list_files(folder, filename, print_count=print_count)
    else:
        list_file_path = io_tools.get_list_files(root_folder, filename_re=filename_re, folder_re=folder_re, print_count=print_count)
    

    if use_dask == None:
        if (auto_dask == True) & (len(list_file_path) >= auto_dask_min_files):
            use_dask = True
        else:
            use_dask = False

    if use_dask==True:
        print(f'Reading {len(list_file_path)} files using dask')
        df = dd.read_parquet(list_file_path, columns=columns, engine=engine).compute()
    else:
        list_df = []
        if progress_bar:            
            for file_path in tqdm(list_file_path, 'reading parquets...'):        
                if sample_frac:
                    list_df.append(pd.read_parquet(file_path, columns=columns, engine=engine).sample(frac=sample_frac, random_state=random_state))
                else:
                    list_df.append(pd.read_parquet(file_path, columns=columns, engine=engine))        
        else:
            for file_path in list_file_path:
                if sample_frac:
                    list_df.append(pd.read_parquet(file_path, columns=columns, engine=engine).sample(frac=sample_frac, random_state=random_state))
                else:
                    list_df.append(pd.read_parquet(file_path, columns=columns, engine=engine))        
        df = pd.concat(list_df)
        
    return df


def convert_dtypes(in_df, in_dict_dtypes, default_dtype=None):
    '''
    Convert dtypes of a dataframe according to given dict of column names and dtypes.
    '''
    in_df = in_df.copy()
    for col_nm in in_df.columns:
        if col_nm in in_dict_dtypes.keys():
            if in_df[col_nm].dtype != in_dict_dtypes[col_nm]:
                in_df[col_nm] = in_df[col_nm].astype(in_dict_dtypes[col_nm])
        elif default_dtype:
            if in_df[col_nm].dtype != default_dtype:
                in_df[col_nm] = in_df[col_nm].astype(default_dtype)
            
    return in_df




def optimize_dtypes(df, excluded_cols=None, only_int=True, allow_unsigned=False):
    
    '''
    Optimize data type of each column to minimum size.
    '''
    df = df.copy()
    
    if excluded_cols:
        assert(type(excluded_cols) == list)
        list_cols = [col for col in df.columns if col not in excluded_cols]
        
    else:
        list_cols = list(df.columns)

 

    
    if (only_int==True) :
        list_cols = [col for col in list_cols if 'int' in str(df[col].dtype)]
        
        
    for col in list_cols:
        col_dtype_ori_str = str(df[col].dtype)        
        col_max_val = df[col].max()
        col_min_val = df[col].min()

 

        if 'int' in col_dtype_ori_str:
            if (col_min_val >= 0) & (allow_unsigned==True):
                if col_max_val < 2**8:
                    col_dtype_new = np.uint8
                elif col_max_val < 2**16:
                    col_dtype_new = np.uint16
                elif col_max_val < 2**32:
                    col_dtype_new = np.uint32
                else:
                    col_dtype_new = np.uint64                    
                    
            else:
                if (col_max_val < 2**7) & (col_min_val >= -2**7):
                    col_dtype_new = np.int8
                elif (col_max_val < 2**15) & (col_min_val >= -2**15):
                    col_dtype_new = np.int16
                elif (col_max_val < 2**31) & (col_min_val >= -2**31):
                    col_dtype_new = np.int32
                else:
                    col_dtype_new = np.int64
                    
            
            assert(col_min_val == col_dtype_new(col_min_val))
            assert(col_max_val == col_dtype_new(col_max_val))            
            col_dtype_new_str = str(col_dtype_new).split("'")[1].split('.')[1]
            if col_dtype_ori_str != col_dtype_new_str:
                df[col] = df[col].astype(col_dtype_new)
                print(f'Column "{col}": {col_dtype_ori_str} -> {col_dtype_new_str}')
        else:
            pass
    
    return df


def write_dict_df_to_excel(dict_df, path_output, engine='openpyxl', print_result=True):
    # Create a Pandas Excel writer using XlsxWriter as the engine
    with pd.ExcelWriter(path_output, engine=engine) as writer:
        # Iterate over the dictionary and save each dataframe to a separate sheet
        for sheet_name, df in dict_df.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    if print_result:
        print(f"Dataframes have been successfully saved to {path_output}")