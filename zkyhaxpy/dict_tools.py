import numpy as np

def map_dict(in_arr, in_dict, in_default=np.nan):
    
    '''
    Mapping an array to a given dict
	'''
    
    
    if type(in_default)==str:        
        out_arr = np.vectorize(in_dict.get)(in_arr, in_default)    
    elif type(in_default) == np.ndarray:
        assert(in_arr.shape==in_default.shape)
        arr_default = in_default        
        out_arr = np.vectorize(in_dict.get)(in_arr, arr_default)
    else:
        arr_default = np.full_like(in_arr, in_default)                
        out_arr = np.vectorize(in_dict.get)(in_arr, arr_default)
    
    return out_arr