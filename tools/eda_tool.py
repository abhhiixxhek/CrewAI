# tools/eda_tool.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas_profiling import ProfileReport
import tempfile
from typing import Tuple
import logging

class EDAAnalyzer:
    """
    Advanced EDA tool with automated report generation and interactive visualizations
    """
    
    def __init__(self, config_path: str = 'config/eda_config.yaml'):
        self.config = self._load_config(config_path)
        self.logger = logging.getLogger(__name__)
        
    def analyze(self, df: pd.DataFrame) -> Tuple[dict, str]:
        report = {
            'summary_stats': self._generate_summary_stats(df),
            'correlation_matrix': self._calculate_correlations(df),
            'visualizations': self._generate_visualizations(df),
            'interactive_report': self._generate_html_report(df)
        }
        return report
    
    def _generate_summary_stats(self, df: pd.DataFrame) -> dict:
        return {
            'describe': df.describe().to_dict(),
            'info': {
                'dtypes': df.dtypes.to_dict(),
                'missing_values': df.isna().sum().to_dict(),
                'unique_counts': df.nunique().to_dict()
            }
        }
    
    def _calculate_correlations(self, df: pd.DataFrame) -> dict:
        numeric_df = df.select_dtypes(include=['number'])
        return {
            'pearson': numeric_df.corr().to_dict(),
            'spearman': numeric_df.corr(method='spearman').to_dict()
        }
    
    def _generate_visualizations(self, df: pd.DataFrame) -> dict:
        viz_paths = {}
        with tempfile.TemporaryDirectory() as tmpdir:
            # Distribution plots
            for col in df.select_dtypes(include=['number']):
                plt.figure()
                sns.histplot(df[col])
                path = f"{tmpdir}/{col}_distribution.png"
                plt.savefig(path)
                viz_paths[f"{col}_distribution"] = path
            
            # Correlation heatmap
            plt.figure(figsize=(12, 8))
            sns.heatmap(df.corr(), annot=True)
            path = f"{tmpdir}/correlation_heatmap.png"
            plt.savefig(path)
            viz_paths['correlation_heatmap'] = path
            
        return viz_paths
    
    def _generate_html_report(self, df: pd.DataFrame) -> str:
        profile = ProfileReport(df, title="Automated EDA Report")
        return profile.to_html()