import logging
import boto3
from botocore.exceptions import ClientError
import os
import re

def upload_file(file_name, bucket, object_name=None):
    """
    
    Upload a file to an S3 bucket    

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    
    (This code is from https://boto3.amazonaws.com/v1/documentation/api/latest/guide/s3-examples.html)
    """

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = os.path.basename(file_name)

    # Upload the file
    s3_client = boto3.client('s3')
    try:
        response = s3_client.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True



def download_file(bucket_name, object_name, path_dest, replace=False):
    if (os.path.exists(path_dest) == True) & (replace==False):
        print(f'{path_dest} already exists. Skipping.')
        return
    s3 = boto3.client('s3')
    s3.download_file(bucket_name, object_name, path_dest)
    

def get_list_of_objects(bucket_name, object_regex='.*'):

    s3 = boto3.client('s3')
    s3_resource = boto3.resource('s3')
    my_bucket = s3_resource.Bucket(bucket_name)

    list_objects = []
    for my_bucket_object in my_bucket.objects.all():
        object_key = my_bucket_object.key
        result = re.match(object_regex, object_key)
        if result:
            list_objects.append(object_key)    