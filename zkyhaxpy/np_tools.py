import numpy as np

def case_when(*args, default=np.nan, dtype=float): 
    '''

    SQL case when condition like for numpy array.

    Parameters
    ----------
    args: a numpy array
        A numpy array of lenght 2 * number of cases. With the first element of a pair is array of boolen and the second element of a pair is value if the condition is matched.
        
    default: int or float
        A value if there is none condition is matched.
    
    dtype: dtype
        dtype of the output array
        

    Examples
    --------
    arr = np.array([0, 2, 5, 3, 4, 10, 4])
    case_when(
        (np.isin(arr, [5, 10]), 5),
        (np.isin(arr, [2, 4, 6, 8, 10]), 2),
        (arr == 0, 0),
        default=-999   
    )

    Returns
    -------
    Numpy array of the same shape as inputted conditions with element assigned according to each case when condition.

	'''
    out_shape = args[0][0].shape
    arr_out = np.full(out_shape, np.nan)

    for i in range(len(args)):
        assert(out_shape == args[i][0].shape)
        arr_out = np.where((np.isnan(arr_out) & (args[i][0])) , args[i][1], arr_out)                
    arr_out = np.where(~np.isnan(arr_out), arr_out, default)
    return arr_out.astype(dtype)

