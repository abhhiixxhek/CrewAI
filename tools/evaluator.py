# tools/evaluator.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (precision_recall_curve, roc_curve, 
                           confusion_matrix, classification_report)
from typing import Tuple, Dict
import logging
import shap
import mlflow
from alibi.explainers import AnchorTabular

class AdvancedModelEvaluator:
    """
    Advanced model evaluation system with explainable AI,
    fairness metrics, and production monitoring
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.shap_explainer = None
        self.anchor_explainer = None

    def evaluate(self, model: Any, X_test: pd.DataFrame, 
                y_test: pd.Series) -> Dict:
        evaluation_report = {
            'metrics': self._calculate_metrics(model, X_test, y_test),
            'visualizations': self._generate_visualizations(model, X_test, y_test),
            'explanations': self._generate_explanations(model, X_test),
            'fairness_report': self._check_fairness(model, X_test, y_test)
        }
        
        mlflow.log_metrics(evaluation_report['metrics'])
        return evaluation_report

    def _calculate_metrics(self, model: Any, X: pd.DataFrame, y: pd.Series) -> Dict:
        preds = model.predict(X)
        probas = model.predict_proba(X)[:,1] if hasattr(model, 'predict_proba') else None
        
        return {
            'classification_report': classification_report(y, preds, output_dict=True),
            'confusion_matrix': confusion_matrix(y, preds).tolist(),
            'roc_auc': roc_auc_score(y, probas) if probas is not None else None,
            'precision_recall_curve': precision_recall_curve(y, probas) if probas else None
        }

    def _generate_visualizations(self, model: Any, X: pd.DataFrame, 
                               y: pd.Series) -> Dict:
        plt.figure(figsize=(10, 6))
        # ROC Curve
        fpr, tpr, _ = roc_curve(y, model.predict_proba(X)[:,1])
        plt.plot(fpr, tpr, label='ROC Curve')
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y, model.predict_proba(X)[:,1])
        plt.plot(recall, precision, label='PR Curve')
        
        plt.legend()
        return {
            'roc_pr_curve': plt.gcf(),
            'feature_importance': self._plot_feature_importance(model, X)
        }

    def _generate_explanations(self, model: Any, X: pd.DataFrame) -> Dict:
        # SHAP Explanations
        self.shap_explainer = shap.TreeExplainer(model)
        shap_values = self.shap_explainer.shap_values(X)
        
        # Anchor Explanations
        self.anchor_explainer = AnchorTabular(
            predict_fn=model.predict_proba,
            feature_names=X.columns.tolist()
        )
        self.anchor_explainer.fit(X)
        
        return {
            'shap_values': shap_values,
            'anchor_examples': self.anchor_explainer.explain(X.sample(3)).anchor
        }

    def _check_fairness(self, model: Any, X: pd.DataFrame, y: pd.Series) -> Dict:
        fairness_report = {}
        for protected_feature in self.config.get('protected_features', []):
            groups = X[protected_feature].unique()
            group_metrics = {}
            
            for group in groups:
                mask = X[protected_feature] == group
                group_metrics[str(group)] = {
                    'accuracy': accuracy_score(y[mask], model.predict(X[mask])),
                    'f1_score': f1_score(y[mask], model.predict(X[mask]))
                }
                
            fairness_report[protected_feature] = group_metrics
        return fairness_report

    def _plot_feature_importance(self, model: Any, X: pd.DataFrame):
        if hasattr(model, 'feature_importances_'):
            importances = pd.Series(model.feature_importances_, index=X.columns)
            plt.figure(figsize=(10, 6))
            importances.sort_values().plot.barh()
            return plt.gcf()