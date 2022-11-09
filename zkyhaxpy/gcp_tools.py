from google.cloud import storage

def get_bucket_and_file_name_from_uri(file_uri):
    assert(file_uri.startswith('gs://'))
    file_uri = file_uri.replace('gs://', '')
    bucket_name = file_uri.split('/')[0]
    file_name = '/'.join(file_uri.split('/')[1:])
    return (bucket_name, file_name)

def check_file_exists_gcs(file_uri):        
    bucket_name, file_name = get_bucket_and_file_name_from_uri(file_uri)
    storage_client = storage.Client()    
    bucket = storage_client.bucket(bucket_name)
    stats = storage.Blob(bucket=bucket, name=file_name).exists(storage_client)
    return stats