# tools/data_loader.py
import pandas as pd
import boto3
from sqlalchemy import create_engine
from io import BytesIO
from typing import Union, Dict
import yaml
import logging

class AdvancedDataLoader:
    """
    Advanced data loader with caching, format auto-detection, and connection pooling
    """
    
    def __init__(self, config_path: str = 'config/data_sources.yaml'):
        self._load_config(config_path)
        self._init_connections()
        self.cache = {}
        logging.basicConfig(level=logging.INFO)

    def _load_config(self, config_path: str):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

    def _init_connections(self):
        self.s3 = boto3.client('s3',
            aws_access_key_id=self.config['s3']['access_key'],
            aws_secret_access_key=self.config['s3']['secret_key']
        )
        self.db_engine = create_engine(self.config['sql']['connection_string'])

    def load(self, source: str) -> pd.DataFrame:
        if source in self.cache:
            return self.cache[source]
            
        if source.startswith('s3://'):
            df = self._load_from_s3(source)
        elif source.startswith('sql://'):
            df = self._load_from_sql(source)
        else:
            df = self._load_from_file(source)
            
        self.cache[source] = df
        return df

    def _load_from_s3(self, s3_uri: str) -> pd.DataFrame:
        bucket, key = s3_uri.replace('s3://', '').split('/', 1)
        response = self.s3.get_object(Bucket=bucket, Key=key)
        file_like = BytesIO(response['Body'].read())
        
        if key.endswith('.parquet'):
            return pd.read_parquet(file_like)
        elif key.endswith('.csv'):
            return pd.read_csv(file_like)
        else:
            raise ValueError(f"Unsupported S3 file format: {key}")

    def _load_from_sql(self, sql_uri: str) -> pd.DataFrame:
        query = sql_uri.replace('sql://', '')
        return pd.read_sql(query, self.db_engine)

    def _load_from_file(self, file_path: str) -> pd.DataFrame:
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        elif file_path.endswith('.parquet'):
            return pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported local file format: {file_path}")