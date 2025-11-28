import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

def upload_to_s3(local_path, bucket, key, region=None, aws_access_key_id=None, aws_secret_access_key=None):
    try:
        import boto3
        from botocore.exceptions import ClientError
    except Exception as e:
        print('boto3 not installed. Install with: pip install boto3')
        raise

    session_kwargs = {}
    if aws_access_key_id and aws_secret_access_key:
        session_kwargs['aws_access_key_id'] = aws_access_key_id
        session_kwargs['aws_secret_access_key'] = aws_secret_access_key
    if region:
        session_kwargs['region_name'] = region


    endpoint_url = os.environ.get('B2_ENDPOINT_URL')
    if endpoint_url:
        s3 = boto3.client('s3', endpoint_url=endpoint_url, **session_kwargs)
    else:
        s3 = boto3.client('s3', **session_kwargs)

    if not os.path.exists(local_path):
        print(f'Local file not found: {local_path}')
        return False

    try:
        print(f"Uploading {local_path} -> s3://{bucket}/{key}")
        s3.upload_file(local_path, bucket, key)
        print('Upload complete')
        return True
    except ClientError as e:
        print('S3 upload failed:', e)
        return False


def upload_chroma(local_path=None):
    """Upload the local Chroma sqlite file to S3.
    Environment variables used:
      - S3_BUCKET_NAME
      - S3_CHROMA_KEY
      - AWS_REGION (optional)
      - AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY (optional)
      - CHROMA_LOCAL_PATH (optional override)
    """
    local_path = local_path or os.environ.get('CHROMA_LOCAL_PATH', 'chroma_data/chroma.sqlite3')
    bucket = os.environ.get('S3_BUCKET_NAME')
    key = os.environ.get('S3_CHROMA_KEY', 'chroma.sqlite3')
    region = os.environ.get('AWS_REGION')
    access_key = os.environ.get('AWS_ACCESS_KEY_ID')
    secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')

    if not bucket:
        print('S3_BUCKET_NAME not set; cannot upload Chroma DB')
        return None

    return upload_to_s3(local_path, bucket, key, region, access_key, secret_key)


if __name__ == '__main__':
    print('Running upload_chroma...')
    result = upload_chroma()
    print('Result:', result)