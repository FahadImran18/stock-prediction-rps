"""
Data loading and storage module
Handles saving to MinIO/S3 and DVC versioning
"""
import os
import logging
from pathlib import Path
from typing import Optional
import pandas as pd
from minio import Minio
from minio.error import S3Error
import boto3
from botocore.exceptions import ClientError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Load data to cloud storage (MinIO/S3)"""
    
    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket: str,
        use_minio: bool = True
    ):
        self.endpoint = endpoint
        self.access_key = access_key
        self.secret_key = secret_key
        self.bucket = bucket
        self.use_minio = use_minio
        
        if use_minio:
            # MinIO client
            self.client = Minio(
                endpoint,
                access_key=access_key,
                secret_key=secret_key,
                secure=False  # Set to True for HTTPS
            )
            self._ensure_bucket_minio()
        else:
            # S3 client
            self.client = boto3.client(
                's3',
                endpoint_url=f"https://{endpoint}",
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key
            )
            self._ensure_bucket_s3()
    
    def _ensure_bucket_minio(self):
        """Ensure MinIO bucket exists"""
        try:
            if not self.client.bucket_exists(self.bucket):
                self.client.make_bucket(self.bucket)
                logger.info(f"Created bucket: {self.bucket}")
            else:
                logger.info(f"Bucket {self.bucket} already exists")
        except S3Error as e:
            logger.error(f"Error ensuring bucket: {e}")
            raise
    
    def _ensure_bucket_s3(self):
        """Ensure S3 bucket exists"""
        try:
            self.client.head_bucket(Bucket=self.bucket)
            logger.info(f"Bucket {self.bucket} exists")
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                self.client.create_bucket(Bucket=self.bucket)
                logger.info(f"Created bucket: {self.bucket}")
            else:
                logger.error(f"Error ensuring bucket: {e}")
                raise
    
    def upload_file(
        self,
        local_filepath: str,
        object_name: Optional[str] = None
    ) -> str:
        """
        Upload file to storage
        
        Args:
            local_filepath: Local file path
            object_name: Object name in storage (default: filename)
            
        Returns:
            Object name/path in storage
        """
        if object_name is None:
            object_name = Path(local_filepath).name
        
        try:
            if self.use_minio:
                self.client.fput_object(
                    self.bucket,
                    object_name,
                    local_filepath
                )
            else:
                self.client.upload_file(
                    local_filepath,
                    self.bucket,
                    object_name
                )
            
            logger.info(f"Uploaded {local_filepath} to {self.bucket}/{object_name}")
            return object_name
            
        except (S3Error, ClientError) as e:
            logger.error(f"Error uploading file: {e}")
            raise


def load_to_storage(
    filepath: str,
    endpoint: str,
    access_key: str,
    secret_key: str,
    bucket: str,
    object_name: Optional[str] = None,
    use_minio: bool = True
) -> str:
    """
    Main loading function for Airflow
    
    Args:
        filepath: Local file path
        endpoint: Storage endpoint
        access_key: Access key
        secret_key: Secret key
        bucket: Bucket name
        object_name: Object name (optional)
        use_minio: Whether to use MinIO (True) or S3 (False)
        
    Returns:
        Object name in storage
    """
    loader = DataLoader(
        endpoint=endpoint,
        access_key=access_key,
        secret_key=secret_key,
        bucket=bucket,
        use_minio=use_minio
    )
    
    return loader.upload_file(filepath, object_name)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        endpoint = os.getenv("MINIO_ENDPOINT", "localhost:9000")
        access_key = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
        secret_key = os.getenv("MINIO_SECRET_KEY", "minioadmin")
        bucket = os.getenv("MINIO_BUCKET", "stock-data")
        
        object_name = load_to_storage(
            filepath,
            endpoint,
            access_key,
            secret_key,
            bucket
        )
        print(f"File uploaded as: {object_name}")
    else:
        print("Usage: python load.py <filepath>")

