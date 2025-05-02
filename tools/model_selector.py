# tools/model_selector.py
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                           roc_auc_score, f1_score, mean_squared_error)
from typing import Dict, Any
import mlflow
import logging
from hyperopt import hp
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import ClassificationPreset

class AdvancedModelSelector:
    """
    Advanced model selection system with statistical testing,
    business metric integration, and drift detection
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.reports = {}
        self.column_mapping = ColumnMapping(
            target=self.config['target_column'],
            prediction='prediction',
            numerical_features=self.config['numerical_features']
        )

    def select_best_model(self, models: Dict, X_val: pd.DataFrame, 
                        y_val: pd.Series) -> Tuple[str, Any]:
        evaluation_results = {}
        
        for model_name, model in models.items():
            metrics = self._evaluate_model(model, X_val, y_val)
            stability = self._check_model_stability(model, X_val, y_val)
            evaluation_results[model_name] = {
                'metrics': metrics,
                'stability': stability,
                'business_impact': self._calculate_business_impact(metrics)
            }
            
        return self._rank_models(evaluation_results)

    def _evaluate_model(self, model: Any, X: pd.DataFrame, y: pd.Series) -> Dict:
        preds = model.predict(X)
        probas = model.predict_proba(X)[:,1] if hasattr(model, 'predict_proba') else None
        
        metrics = {
            'accuracy': accuracy_score(y, preds),
            'precision': precision_score(y, preds),
            'recall': recall_score(y, preds),
            'roc_auc': roc_auc_score(y, probas) if probas is not None else None,
            'f1': f1_score(y, preds)
        }
        
        # Generate evident report
        report = Report(metrics=[ClassificationPreset()])
        report.run(reference_data=None, 
                 current_data=pd.concat([X, y], axis=1),
                 column_mapping=self.column_mapping)
        
        self.reports[model.__class__.__name__] = report
        return metrics

    def _check_model_stability(self, model: Any, X: pd.DataFrame, 
                             y: pd.Series) -> Dict:
        bootstrap_scores = []
        for _ in range(100):
            sample_idx = np.random.choice(X.index, size=len(X), replace=True)
            X_sample = X.loc[sample_idx]
            y_sample = y.loc[sample_idx]
            score = model.score(X_sample, y_sample)
            bootstrap_scores.append(score)
            
        return {
            'mean_score': np.mean(bootstrap_scores),
            'std_dev': np.std(bootstrap_scores),
            'confidence_interval': np.percentile(bootstrap_scores, [2.5, 97.5])
        }

    def _calculate_business_impact(self, metrics: Dict) -> Dict:
        # Example business impact calculation
        cost_matrix = self.config.get('cost_matrix', {})
        expected_loss = (
            metrics['recall'] * cost_matrix['fn_cost'] +
            (1 - metrics['precision']) * cost_matrix['fp_cost']
        )
        return {'expected_loss': expected_loss}

    def _rank_models(self, results: Dict) -> Tuple[str, Any]:
        return max(results.items(), key=lambda x: x[1]['business_impact']['expected_loss'])