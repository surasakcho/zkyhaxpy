import pandas as pd
from IPython.display import display, HTML, display_html

def auto_adjust():
    set_colwidth()
    pd.options.display.max_rows = 999
    pd.options.display.max_columns = 100


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


def curr_colwidth():
    '''
    Decrease column width of pandas dataframe display
    
    Return : None
    '''

    print(f'Current max column width = {pd.get_option("display.max_colwidth")}')