# tools/model_trainer.py
import numpy as np
import pandas as pd
from sklearn.model_selection import (cross_val_score, StratifiedKFold, 
                                   RandomizedSearchCV)
from sklearn.metrics import get_scorer
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                            HistGradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from typing import Dict, Tuple, Any
import mlflow
import logging
import joblib
import json
from hyperopt import fmin, tpe, hp, STATUS_OK

class AdvancedModelTrainer:
    """
    Advanced model trainer with automated hyperparameter tuning,
    cross-validation, and experiment tracking
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.best_models = {}
        self.mlflow_uri = self.config.get('mlflow_uri', 'http://localhost:5000')

    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        mlflow.set_tracking_uri(self.mlflow_uri)
        mlflow.set_experiment(self.config['experiment_name'])

        results = {}
        for model_name, params in self.config['models'].items():
            with mlflow.start_run(run_name=model_name):
                model = self._init_model(model_name)
                scores = self._cross_validate(model, X, y)
                
                if self.config.get('hyperparameter_tuning'):
                    model = self._optimize_hyperparameters(model, params, X, y)
                
                self.best_models[model_name] = model
                results[model_name] = scores
                
                mlflow.log_params(model.get_params())
                mlflow.log_metrics(scores)
                mlflow.sklearn.log_model(model, model_name)
        
        return results

    def _init_model(self, model_name: str) -> Any:
        models = {
            'random_forest': RandomForestClassifier(),
            'xgboost': XGBClassifier(),
            'lightgbm': LGBMClassifier(),
            'logistic_regression': LogisticRegression()
        }
        return models[model_name]

    def _cross_validate(self, model: Any, X: pd.DataFrame, y: pd.Series) -> Dict:
        cv = StratifiedKFold(n_splits=5)
        scores = {}
        
        for metric in self.config['metrics']:
            scorer = get_scorer(metric)
            cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scorer)
            scores[metric] = np.mean(cv_scores)
            
        return scores

    def _optimize_hyperparameters(self, model: Any, space: Dict, 
                                X: pd.DataFrame, y: pd.Series) -> Any:
        def objective(params):
            model.set_params(**params)
            score = cross_val_score(model, X, y, 
                                  cv=StratifiedKFold(3),
                                  scoring=self.config['optimization_metric']).mean()
            return {'loss': -score, 'status': STATUS_OK}
            
        best_params = fmin(fn=objective,
                         space=space,
                         algo=tpe.suggest,
                         max_evals=100)
        model.set_params(**best_params)
        return model.fit(X, y)

    def save_model(self, model_name: str, path: str):
        joblib.dump(self.best_models[model_name], path)
        mlflow.sklearn.save_model(self.best_models[model_name], path)