from skopt.space import Real, Integer, Categorical

def get_skopt_search_space(in_dict_search_space):
    '''
    Transform dict of search space into skopt search space formatted.
    
    Input dict must be formatted as follow
    
    in_dict_search_space = {
        "param_1":{
            "type":"int",
            "min":0,
            "max":10,
            "scale":"auto"
        },
        "param_2":{
            "type":"real",
            "min":0.1,
            "max":100,
            "scale":"log"
        },
        "param_3":{
            "type":"categorical",
            "values":["a", "b", "c"]
        }
    }
    
    '''
    
    out_dict_skopt_search_space = {}
    for parm_nm in in_dict_search_space.keys():
        if in_dict_search_space[parm_nm].get('scale')=='auto':
            scale = 'uniform'
        elif in_dict_search_space[parm_nm].get('scale')=='log':
            scale = 'log-uniform'
        else:
            scale = 'auto'
            
        if in_dict_search_space[parm_nm]['type']=='real':
            out_dict_skopt_search_space[parm_nm] = Real(
                low=in_dict_search_space[parm_nm]['min'],
                high=in_dict_search_space[parm_nm]['max'],
                prior=scale
            )
        elif in_dict_search_space[parm_nm]['type']=='int':
            out_dict_skopt_search_space[parm_nm] = Integer(
                low=in_dict_search_space[parm_nm]['min'],
                high=in_dict_search_space[parm_nm]['max'],
                prior=scale
            )
        elif in_dict_search_space[parm_nm]['type']=='categorical':
            out_dict_skopt_search_space[parm_nm] = Categorical(categories=in_dict_search_space[parm_nm]['values'], prior=None)
        
    return out_dict_skopt_search_space
