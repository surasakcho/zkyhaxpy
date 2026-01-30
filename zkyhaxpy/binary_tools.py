import base64
import math

def read_binary_to_chunks(file_path, num_chunks=10):
    """
    Reads a binary file, converts to Base64, and splits 
    it into exactly 'num_chunks' (or fewer for tiny files).
    """
    with open(file_path, 'rb') as f:
        binary_data = f.read()
    
    base64_str = base64.b64encode(binary_data).decode('utf-8')
    total_len = len(base64_str)
    
    # Calculate how many characters per chunk
    # We use ceil to ensure we don't end up with an extra tiny chunk at the end
    chunk_size = math.ceil(total_len / num_chunks)
    
    # Handle edge case for empty files
    if chunk_size == 0:
        return [base64_str]
        
    return [base64_str[i:i + chunk_size] for i in range(0, total_len, chunk_size)]

def write_chunks_to_binary(file_path, chunks):
    """Reassembles the chunks and saves as binary."""
    full_base64_str = "".join(chunks)
    binary_data = base64.b64decode(full_base64_str.encode('utf-8'))
    
    with open(file_path, 'wb') as f:
        f.write(binary_data)