import datetime
import re
import numpy as np

dict_th_month = {
        'มกราคม':1,
        'กุมภาพันธ์':2,
        'มีนาคม':3,
        'เมษายน':4,
        'พฤษภาคม':5,
        'มิถุนายน':6,
        'กรกฎาคม':7,
        'สิงหาคม':8,
        'กันยายน':9,
        'ตุลาคม':10,
        'พฤศจิกายน':11,
        'ธันวาคม':12,
        'ม.ค.':1,
        'ก.พ.':2,
        'มี.ค.':3,
        'เม.ย.':4,
        'พ.ค.':5,
        'มิ.ย.':6,
        'ก.ค.':7,
        'ส.ค.':8,
        'ก.ย.':9,
        'ต.ค.':10,
        'พ.ย.':11,
        'ธ.ค.':12,
    }
    
    
def dttm_to_yyyymmdd(dttm):
    '''
    Return : a string of current datetime in yyyymmdd_hhmm 
    '''
    
    if type(dttm) is np.datetime64:
        return ''.join(str(np.datetime64(dttm))[:10].split('-'))
    elif (type(dttm) is datetime.date) | (type(dttm) is datetime.datetime):
        return datetime.datetime.strftime(dttm, '%Y%m%d')



def curr_yyyymmdd_hhmm():
    '''
    Return : a string of current datetime in yyyymmdd_hhmm 
    '''
    return get_curr_yyyymmdd_hhmm()

def get_curr_yyyymmdd_hhmm():
    '''
    Return : a string of current datetime in yyyymmdd_hhmm 
    '''
    return datetime.datetime.strftime(datetime.datetime.today(), '%Y%m%d_%H%M')


def get_curr_yyyymmdd():
    '''
    Return : a string of current date in yyyymmdd
    '''
    return datetime.datetime.strftime(datetime.datetime.today(), '%Y%m%d')    


def curr_yyyymmdd():
    '''
    Return : a string of current date in yyyymmdd
    '''
    return get_curr_yyyymmdd()


def get_days_diff_yyyymmdd(yyyymmdd_1, yyyymmdd_2):
    '''
    Return : no. of days different between 2 dates in yyyymmdd format
    '''
    date_1 = datetime.datetime.strptime(yyyymmdd_1, '%Y%m%d')
    date_2 = datetime.datetime.strptime(yyyymmdd_2, '%Y%m%d')
    days_diff = date_1 - date_2
    return days_diff.days    


def days_diff_yyyymmdd(yyyymmdd_1, yyyymmdd_2):
    '''
    Return : no. of days different between 2 dates in yyyymmdd format
    '''

    return get_days_diff_yyyymmdd(yyyymmdd_1, yyyymmdd_2)  


def yyyymmdd_add_days(yyyymmdd, days):
    '''
    Return : no. of days different between 2 dates in yyyymmdd format
    '''
    date = datetime.datetime.strptime(yyyymmdd, '%Y%m%d') + datetime.timedelta(days=days)
    return datetime.datetime.strftime(date, '%Y%m%d') 


def convert_th_date_to_datetime(str_th_date):
    '''

    Convert a string of Thai date into datetime.

    Parameters
    ----------
    str_th_date: str
        A string of Thai date (ex. 1 ก.ค. 56, 1 มีนาคม 2563)
    
    Returns
    -------
    out_datetime : datetime
    
	'''

    for str_month, int_month in dict_th_month.items():
        str_th_date = str_th_date.replace(str_month, str(int_month))

    str_th_date = str_th_date.replace(' ', '-')

    int_day = int(re.findall('^\\d{1,2}-', str_th_date)[0][:-1])
    
    int_month = int(re.findall('-\\d{1,2}-', str_th_date)[0][1:-1])  
        
    str_year = re.findall('-\\d{2,4}$', str_th_date)[0][1:]

    #year
    if len(str_year) == 2:
        int_year = int(str_year) - 543 + 2500
    elif len(str_year) == 4:
        int_year = int(str_year) - 543
    else:
        int_year = 1900    
    
    out_datetime = datetime.datetime(year=int_year, month=int_month, day=int_day)
    return out_datetime



def date_to_date_ndays(arr_date, date_of_day_one = '2000-01-01', nbr_days_per_period=16):
    '''
    Get array of  N days of date
    '''

    arr_datediff_from_day_one = (arr_date - np.datetime64(date_of_day_one)) / np.timedelta64(1, 'D')
    arr_date_ndays = np.datetime64(date_of_day_one) + np.ceil(arr_datediff_from_day_one / nbr_days_per_period) * nbr_days_per_period * np.timedelta64(1, 'D')
    return arr_date_ndays    




def yyyymmdd_to_numpy_datetime(arr_yyyymmdd):
    '''
    Convert array of string yyyymmdd into array of numpy datetime64
    '''
    
    if arr_yyyymmdd.dtype != int:
        arr_yyyymmdd = arr_yyyymmdd.astype(int)
  
    arr_yyyy = np.floor(arr_yyyymmdd / 10000)
    arr_mm = np.floor(arr_yyyymmdd / 100) % 100
    arr_dd = arr_yyyymmdd % 100

    arr_yyyy = arr_yyyy.astype(int).astype(str)
    arr_mm = arr_mm.astype(int).astype(str)
    arr_dd = arr_dd.astype(int).astype(str)

    arr_mm = np.char.zfill(arr_mm, 2)
    arr_dd = np.char.zfill(arr_dd, 2)
    arr_datetime_str = arr_yyyy
    arr_datetime_str = np.core.defchararray.add(arr_datetime_str, '-')
    arr_datetime_str = np.core.defchararray.add(arr_datetime_str, arr_mm)
    arr_datetime_str = np.core.defchararray.add(arr_datetime_str, '-')
    arr_datetime_str = np.core.defchararray.add(arr_datetime_str, arr_dd)

    arr_datetime = arr_datetime_str.astype(dtype='datetime64[D]')

    return arr_datetime