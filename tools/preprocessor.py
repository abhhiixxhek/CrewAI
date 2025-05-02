# tools/preprocessor.py
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any

class AdvancedPreprocessor(TransformerMixin, BaseEstimator):
    """
    Advanced preprocessing pipeline with automated type detection and schema validation
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.imputers = {}
        self.scalers = {}
        self.logger = logging.getLogger(__name__)
        
    def fit(self, X: pd.DataFrame, y=None):
        self._detect_schema(X)
        self._create_pipeline_components()
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        self._validate_schema(X)
        
        # Handle missing values
        for col, strategy in self.config['missing_values'].items():
            if col in X.columns:
                if strategy == 'auto':
                    strategy = 'median' if np.issubdtype(X[col].dtype, np.number) else 'most_frequent'
                
                imputer = SimpleImputer(strategy=strategy)
                X[col] = imputer.fit_transform(X[[col]]).ravel()
        
        # Handle outliers
        if self.config.get('handle_outliers', False):
            numeric_cols = X.select_dtypes(include=np.number).columns
            scaler = RobustScaler()
            X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
        
        # Type casting
        for col, dtype in self.config.get('dtype_casts', {}).items():
            if col in X.columns:
                X[col] = X[col].astype(dtype)
        
        self.logger.info("Preprocessing completed successfully")
        return X

    def _detect_schema(self, X: pd.DataFrame):
        self.schema = {
            'columns': list(X.columns),
            'dtypes': X.dtypes.to_dict(),
            'numeric_cols': X.select_dtypes(include=np.number).columns.tolist(),
            'categorical_cols': X.select_dtypes(exclude=np.number).columns.tolist()
        }

    def _validate_schema(self, X: pd.DataFrame):
        if set(X.columns) != set(self.schema['columns']):
            raise ValueError("Schema mismatch between fitted and transformed data")