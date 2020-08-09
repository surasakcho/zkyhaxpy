import subprocess

def execute_cmd(cmd, return_output_line=False):
    '''

    Execute a command on OS' console.
    
    Parameters
    ----------
    cmd: str
        A string of command to be executed on OS' console.
        
    return_output_line : boolean
        If True, return a list of output lines.
    
    Returns
    -------
    If return_output_line is True, return a list of output lines.

	'''
    print(f'Executing "{cmd}" on console.')
    
    list_output = []
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, shell=True)
    
    for line in iter(p.stdout.readline, b''):
        line_str = line.decode("utf-8") 
        list_output.append(line_str)
        print(line_str)
    p.stdout.close()
    p.wait()
    
    if return_output_line==True:
        return list_output
    