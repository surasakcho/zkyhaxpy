# python3
import hashlib


def encrypt_string(input_string, salt):
    # Assume cleaned input string

    # Append salt to the end of input string
    input_string = input_string + salt

    # encrypt using utf-8 encoding
    encrypted_string = hashlib.sha256(input_string.encode('utf8')).hexdigest()

    return encrypted_string


