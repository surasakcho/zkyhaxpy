
import json
import os
import numpy as np


def read_dict_json(json_path, print_result=False, convert_non_allowed_keys=True):   
    '''
    Read a json file into a dictionary.
    '''
    with open(json_path, 'r') as f:
        dict_out = json.load(f)
        
    #Convert non allowed keys back to original key
    if convert_non_allowed_keys==True:
        dict_out = __convert_temp_keys_to_original(dict_out)    
    
    if print_result:    
        print(f'{json_path} has been loaded.')
    return dict_out


def write_dict_json(dict_in, json_path, force_overwrite=True, convert_non_allowed_keys=True, print_result=True, indent=2, auto_clean=True):   

    '''
    Write a dictionary into a json file.
    '''    
    
    #Check if the output file already exists and force_overwrite is false
    if (os.path.exists(json_path)==True) & (force_overwrite==False):
        str_input = input(f'"{json_path}" already exists.\nDo you want to overwrite it? (YES/NO)')
        if (str_input != 'YES'):
            #Do nothing
            print(f'Saving {json_path} has been cancelled.')
            return 
    
    #Clean dict before writing out
    if auto_clean:
        dict_in = clean_dict_for_exporting(dict_in)

    #Convert non allowed keys back to original key
    if convert_non_allowed_keys==True:
        dict_in = __convert_non_allowed_keys_to_temp(dict_in)
    
    with open(json_path, 'w') as f:
        f.write(json.dumps(dict_in, indent=indent))
        if print_result:
            print(f'{json_path} has been saved.')
    return 


def __convert_non_allowed_keys_to_temp(dict_in):
    '''
    An internal process for converting non allowed keys for json file into temp keys.
    '''
    dict_out = {}
    dict_out['TEMP_KEYS'] = {}
    for key in dict_in.keys():
        if type(key) != str:  
            temp_key = str(key)
            if type(key) == tuple:
                dict_out['TEMP_KEYS'].update({temp_key:list(key)})
            else:
                dict_out['TEMP_KEYS'].update({temp_key:key})
            
            if type(dict_in[key]) == dict:       
                dict_out[temp_key] = __convert_non_allowed_keys_to_temp(dict_in[key])
            else:
                dict_out[temp_key] = dict_in[key]
        else:
            
            if type(dict_in[key]) == dict:       
                dict_out[key] = __convert_non_allowed_keys_to_temp(dict_in[key])
            else:
                dict_out[key] = dict_in[key]
        
    return dict_out
    
    
def __convert_temp_keys_to_original(dict_in):
    '''
    An internal process for converting temp keys into their original keys.
    '''
    dict_out = {}
    if 'TEMP_KEYS' in dict_in.keys():
        dict_temp_keys = dict_in['TEMP_KEYS']
    else:
        dict_temp_keys = {}
    
    for key in dict_in.keys():        
        if key != 'TEMP_KEYS':            
            if key in dict_temp_keys.keys():
                original_key = dict_temp_keys[key]
                if type(original_key) == list:
                    original_key = tuple(original_key)
                if type(dict_in[key]) == dict:
                    dict_out[original_key] = __convert_temp_keys_to_original(dict_in[key])
                else:
                    dict_out[original_key] = dict_in[key]                                    
            else:                
                if type(dict_in[key]) == dict:
                    dict_out[key] = __convert_temp_keys_to_original(dict_in[key])
                else:
                    dict_out[key] = dict_in[key]                                        
    return dict_out
        
    
def clean_dict_for_exporting(dict_input):
    dict_output = dict_input.copy()
    for key in dict_output.keys():
        try:
            if dict_output[key] == int(dict_output[key]):
                dict_output[key] = int(dict_output[key])
            else:
                if type(dict_output[key]) == np.float32:
                    dict_output[key] = float(dict_output[key])

                    
        except ValueError:
            pass
        except OverflowError:
            if dict_output[key] > 0:
                dict_output[key] = 999999999
            else:
                dict_output[key] = -999999999
    return dict_output    