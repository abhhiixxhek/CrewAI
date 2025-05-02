# tools/feature_engineering.py
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import (StandardScaler, OneHotEncoder, 
                                   PolynomialFeatures, PowerTransformer)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import (RFECV, SelectFromModel, 
                                      mutual_info_classif)
import featuretools as ft
from xgboost import XGBClassifier
from typing import List, Dict, Union
import logging
import joblib
import json

class AdvancedFeatureEngineer(TransformerMixin, BaseEstimator):
    """
    Advanced feature engineering pipeline with automated feature creation,
    selection, and schema persistence
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.feature_pipeline = None
        self.feature_names = []
        self.schema = {}

    def fit(self, X: pd.DataFrame, y=None):
        self._validate_input(X)
        self._create_feature_pipeline(X)
        self._fit_feature_tools(X, y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._validate_schema(X)
        
        # Apply automated feature engineering
        if 'feature_tools' in self.config:
            X = self._apply_feature_tools(X)
            
        # Apply sklearn transformations
        if self.feature_pipeline:
            X_transformed = self.feature_pipeline.transform(X)
            if isinstance(X_transformed, np.ndarray):
                X = pd.DataFrame(X_transformed, 
                                columns=self.feature_pipeline.get_feature_names_out())
                
        # Apply feature selection
        if self.config.get('feature_selection'):
            X = self._apply_feature_selection(X)
            
        return X

    def _create_feature_pipeline(self, X: pd.DataFrame):
        numeric_features = X.select_dtypes(include=np.number).columns.tolist()
        categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=2, include_bias=False)),
            ('power', PowerTransformer())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        self.feature_pipeline = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
            
        self.feature_pipeline.fit(X)

    def _apply_feature_tools(self, X: pd.DataFrame):
        es = ft.EntitySet(id='auto_features')
        es = es.entity_from_dataframe(entity_id='data', 
                                    dataframe=X, 
                                    index='auto_index')
        
        feature_matrix, features = ft.dfs(entityset=es,
                                        target_entity='data',
                                        max_depth=2,
                                        verbose=True)
        return feature_matrix

    def _apply_feature_selection(self, X: pd.DataFrame):
        selector = SelectFromModel(
            XGBClassifier(n_estimators=100),
            threshold="median"
        )
        return selector.fit_transform(X, y)

    def save_pipeline(self, path: str):
        joblib.dump(self.feature_pipeline, path)
        with open(f"{path}_schema.json", 'w') as f:
            json.dump(self.schema, f)

    def load_pipeline(self, path: str):
        self.feature_pipeline = joblib.load(path)
        with open(f"{path}_schema.json", 'r') as f:
            self.schema = json.load(f)