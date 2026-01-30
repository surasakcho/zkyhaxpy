import os

def list_data_package_files(full_path=True):
    """
    Returns a list of all asset files bundled in the package.
    If full_path is True, returns the absolute path to each file.
    """
    # Locates the zkyhaxpy folder on the current system
    root = os.path.dirname(__file__)
    data_path = os.path.join(root, 'data_package')
    
    if not os.path.exists(data_path):
        return []

    files = os.listdir(data_path)
    
    if full_path:
        # Join the data_path with each filename to get the absolute path
        return [os.path.abspath(os.path.join(data_path, f)) for f in files]
    
    return files

