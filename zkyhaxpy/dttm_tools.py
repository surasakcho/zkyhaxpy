import datetime

def dttm_to_yyyymmdd(dttm):
    '''
    Return : a string of current datetime in yyyymmdd_hhmm 
    '''
    return datetime.datetime.strftime(dttm, '%Y%m%d')


def curr_yyyymmdd_hhmm():
    '''
    Return : a string of current datetime in yyyymmdd_hhmm 
    '''
    return datetime.datetime.strftime(datetime.datetime.today(), '%Y%m%d_%H%M')

def curr_yyyymmdd():
    '''
    Return : a string of current date in yyyymmdd
    '''
    return datetime.datetime.strftime(datetime.datetime.today(), '%Y%m%d')    

def days_diff_yyyymmdd(yyyymmdd_1, yyyymmdd_2):
    '''
    Return : no. of days different between 2 dates in yyyymmdd format
    '''
    date_1 = datetime.datetime.strptime(yyyymmdd_1, '%Y%m%d')
    date_2 = datetime.datetime.strptime(yyyymmdd_2, '%Y%m%d')
    days_diff = date_1 - date_2
    return days_diff.days    

def yyyymmdd_add_days(yyyymmdd, days):
    '''
    Return : no. of days different between 2 dates in yyyymmdd format
    '''
    date = datetime.datetime.strptime(yyyymmdd, '%Y%m%d') + datetime.timedelta(days=days)
    return datetime.datetime.strftime(date, '%Y%m%d') 