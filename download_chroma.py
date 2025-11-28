import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

def download_from_s3(bucket, key, dest_path, region=None, aws_access_key_id=None, aws_secret_access_key=None):
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

    dest_dir = Path(dest_path).parent
    dest_dir.mkdir(parents=True, exist_ok=True)

    try:
        print(f"Downloading s3://{bucket}/{key} -> {dest_path}")
        s3.download_file(bucket, key, str(dest_path))
        print('Download complete')
        return True
    except ClientError as e:
        print('S3 download failed:', e)
        return False


def ensure_chroma_local(local_path=None):
    """Ensure the Chroma sqlite file exists locally. If not, try to download it from S3.
    Environment variables used:
      - S3_BUCKET_NAME
      - S3_CHROMA_KEY
      - AWS_REGION (optional)
      - AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY (optional)
      - CHROMA_LOCAL_PATH (optional override)
    """
    local_path = local_path or os.environ.get('CHROMA_LOCAL_PATH', 'chroma_data/chroma.sqlite3')
    if os.path.exists(local_path):
        print('Chroma file already exists at', local_path)
        return local_path

    bucket = os.environ.get('S3_BUCKET_NAME')
    key = os.environ.get('S3_CHROMA_KEY', 'chroma.sqlite3')
    region = os.environ.get('AWS_REGION')
    access_key = os.environ.get('AWS_ACCESS_KEY_ID')
    secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')

    if not bucket:
        print('S3_BUCKET_NAME not set; cannot download Chroma DB')
        return None

    success = download_from_s3(bucket, key, local_path, region, access_key, secret_key)
    if success:
        return local_path
    return None

if __name__ == '__main__':
    print('Running download_chroma ensure...')
    p = ensure_chroma_local()
    print('Result:', p)
