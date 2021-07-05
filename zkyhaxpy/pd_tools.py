import pandas as pd
from IPython.display import display, HTML, display_html

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
    
    



        
def read_parquets(list_file_path, columns='all'):
    '''

    Read multiple parquet files of the same template into a single pandas dataframe.

	'''
    
    list_df = []
    for file_path in list_file_path:
        if columns=='all':
            list_df.append(pd.read_parquet(file_path))
        else:
            list_df.append(pd.read_parquet(file_path, columns=columns))
        
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