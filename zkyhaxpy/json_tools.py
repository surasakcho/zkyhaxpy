
import json
import os


def read_dict_json(json_path):   
    '''
    Read a json file into a dictionary.
    '''
    with open(json_path, 'r') as f:
        out_dict = json.load(f)
    print(f'{json_path} has been loaded.')
    return out_dict


def write_dict_json(in_dict, json_path, force_overwrite=False):   
    '''
    Write a dictionary into a json file.
    '''    

    #Check if the output file already exists and force_overwrite is false
    if (os.path.exists(json_path)==True) & (force_overwrite==False):
        str_input = input(f'"{json_path}" already exists.\nDo you want to overwrite it? (YES/NO)')
        if (str_input != 'YES'):
            #Do nothing
            return 
            
    with open(json_path, 'w') as f:
        f.write(json.dumps(in_dict, indent=2))
        print(f'{json_path} has been saved.')
    return 