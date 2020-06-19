def replace_nonchar(in_str, list_old_nonchar=[':', '(', ')', ',', '_'], new_nonchar='-' ):
    out_str = in_str
    
    for old_nonchar in list_old_nonchar:
        out_str = out_str.replace(old_nonchar, ' ')

    out_str = out_str.strip()
    out_str = out_str.replace(' ', new_nonchar)
    while f'{new_nonchar}{new_nonchar}' in out_str:
        out_str = out_str.replace(f'{new_nonchar}{new_nonchar}', new_nonchar)
    
    return out_str


    