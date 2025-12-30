"""Feature engineering and screening module with PCC, RFE analysis."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Optional, Dict
from sklearn.feature_selection import RFE
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr


class FeatureSelector:
    """
    Feature selection and engineering using various methods including
    Pearson Correlation Coefficient (PCC) and Recursive Feature Elimination (RFE).
    """
    
    def __init__(self):
        """Initialize the FeatureSelector."""
        self.selected_features = None
        self.feature_importance = None
        self.correlation_matrix = None
        
    def calculate_pcc(self, X: np.ndarray, y: np.ndarray, 
                      feature_names: List[str]) -> pd.DataFrame:
        """
        Calculate Pearson Correlation Coefficients between features and target.
        
        Args:
            X: Feature matrix
            y: Target vector
            feature_names: Names of features
            
        Returns:
            DataFrame with correlation coefficients and p-values
        """
        correlations = []
        p_values = []
        
        for i in range(X.shape[1]):
            corr, p_val = pearsonr(X[:, i], y)
            correlations.append(corr)
            p_values.append(p_val)
        
        pcc_df = pd.DataFrame({
            'feature': feature_names,
            'correlation': correlations,
            'p_value': p_values,
            'abs_correlation': np.abs(correlations)
        })
        
        pcc_df = pcc_df.sort_values('abs_correlation', ascending=False)
        
        return pcc_df
    
    def plot_correlation_matrix(self, X: np.ndarray, y: np.ndarray,
                               feature_names: List[str],
                               figsize: Tuple[int, int] = (12, 10),
                               save_path: Optional[str] = None):
        """
        Plot correlation matrix heatmap.
        
        Args:
            X: Feature matrix
            y: Target vector
            feature_names: Names of features
            figsize: Figure size
            save_path: Path to save the plot
        """
        # Create DataFrame with features and target
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y
        
        # Calculate correlation matrix
        self.correlation_matrix = df.corr()
        
        # Plot heatmap
        plt.figure(figsize=figsize)
        sns.heatmap(self.correlation_matrix, annot=True, fmt='.2f', 
                   cmap='coolwarm', center=0, square=True)
        plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def rfe_ridge(self, X: np.ndarray, y: np.ndarray, 
                  feature_names: List[str],
                  n_features_to_select: int = 10,
                  alpha: float = 1.0) -> Dict:
        """
        Recursive Feature Elimination with Ridge regression.
        
        Args:
            X: Feature matrix
            y: Target vector
            feature_names: Names of features
            n_features_to_select: Number of features to select
            alpha: Regularization strength for Ridge
            
        Returns:
            Dictionary with selected features and rankings
        """
        ridge = Ridge(alpha=alpha, random_state=42)
        rfe = RFE(estimator=ridge, n_features_to_select=n_features_to_select)
        rfe.fit(X, y)
        
        selected_mask = rfe.support_
        rankings = rfe.ranking_
        
        results = pd.DataFrame({
            'feature': feature_names,
            'selected': selected_mask,
            'ranking': rankings
        })
        results = results.sort_values('ranking')
        
        return {
            'model': 'Ridge',
            'selected_features': [f for f, s in zip(feature_names, selected_mask) if s],
            'rankings': results,
            'rfe_object': rfe
        }
    
    def rfe_huber(self, X: np.ndarray, y: np.ndarray,
                  feature_names: List[str],
                  n_features_to_select: int = 10,
                  epsilon: float = 1.35) -> Dict:
        """
        Recursive Feature Elimination with Huber regression.
        
        Args:
            X: Feature matrix
            y: Target vector
            feature_names: Names of features
            n_features_to_select: Number of features to select
            epsilon: Epsilon parameter for Huber regression
            
        Returns:
            Dictionary with selected features and rankings
        """
        huber = HuberRegressor(epsilon=epsilon, max_iter=200)
        rfe = RFE(estimator=huber, n_features_to_select=n_features_to_select)
        rfe.fit(X, y)
        
        selected_mask = rfe.support_
        rankings = rfe.ranking_
        
        results = pd.DataFrame({
            'feature': feature_names,
            'selected': selected_mask,
            'ranking': rankings
        })
        results = results.sort_values('ranking')
        
        return {
            'model': 'Huber',
            'selected_features': [f for f, s in zip(feature_names, selected_mask) if s],
            'rankings': results,
            'rfe_object': rfe
        }
    
    def rfe_random_forest(self, X: np.ndarray, y: np.ndarray,
                         feature_names: List[str],
                         n_features_to_select: int = 10,
                         n_estimators: int = 100) -> Dict:
        """
        Recursive Feature Elimination with Random Forest.
        
        Args:
            X: Feature matrix
            y: Target vector
            feature_names: Names of features
            n_features_to_select: Number of features to select
            n_estimators: Number of trees in the forest
            
        Returns:
            Dictionary with selected features and rankings
        """
        rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
        rfe = RFE(estimator=rf, n_features_to_select=n_features_to_select)
        rfe.fit(X, y)
        
        selected_mask = rfe.support_
        rankings = rfe.ranking_
        
        results = pd.DataFrame({
            'feature': feature_names,
            'selected': selected_mask,
            'ranking': rankings
        })
        results = results.sort_values('ranking')
        
        return {
            'model': 'RandomForest',
            'selected_features': [f for f, s in zip(feature_names, selected_mask) if s],
            'rankings': results,
            'rfe_object': rfe
        }
    
    def compare_rfe_methods(self, X: np.ndarray, y: np.ndarray,
                           feature_names: List[str],
                           n_features_to_select: int = 10) -> pd.DataFrame:
        """
        Compare RFE results from Ridge, Huber, and Random Forest.
        
        Args:
            X: Feature matrix
            y: Target vector
            feature_names: Names of features
            n_features_to_select: Number of features to select
            
        Returns:
            DataFrame comparing feature selection across methods
        """
        # Run all three RFE methods
        ridge_results = self.rfe_ridge(X, y, feature_names, n_features_to_select)
        huber_results = self.rfe_huber(X, y, feature_names, n_features_to_select)
        rf_results = self.rfe_random_forest(X, y, feature_names, n_features_to_select)
        
        # Create comparison DataFrame
        comparison = pd.DataFrame({
            'feature': feature_names,
            'ridge_selected': [f in ridge_results['selected_features'] for f in feature_names],
            'huber_selected': [f in huber_results['selected_features'] for f in feature_names],
            'rf_selected': [f in rf_results['selected_features'] for f in feature_names]
        })
        
        # Add vote count
        comparison['vote_count'] = (comparison['ridge_selected'].astype(int) + 
                                   comparison['huber_selected'].astype(int) + 
                                   comparison['rf_selected'].astype(int))
        
        comparison = comparison.sort_values('vote_count', ascending=False)
        
        return comparison
    
    def plot_feature_importance(self, pcc_df: pd.DataFrame,
                               top_n: int = 20,
                               figsize: Tuple[int, int] = (10, 8),
                               save_path: Optional[str] = None):
        """
        Plot feature importance based on PCC.
        
        Args:
            pcc_df: DataFrame with PCC results
            top_n: Number of top features to display
            figsize: Figure size
            save_path: Path to save the plot
        """
        top_features = pcc_df.head(top_n)
        
        plt.figure(figsize=figsize)
        colors = ['green' if x > 0 else 'red' for x in top_features['correlation']]
        plt.barh(range(len(top_features)), top_features['correlation'], color=colors)
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Pearson Correlation Coefficient', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.title(f'Top {top_n} Features by Correlation with Target', 
                 fontsize=14, fontweight='bold')
        plt.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def select_features_by_threshold(self, pcc_df: pd.DataFrame, 
                                    threshold: float = 0.3) -> List[str]:
        """
        Select features based on correlation threshold.
        
        Args:
            pcc_df: DataFrame with PCC results
            threshold: Minimum absolute correlation threshold
            
        Returns:
            List of selected feature names
        """
        selected = pcc_df[pcc_df['abs_correlation'] >= threshold]['feature'].tolist()
        self.selected_features = selected
        return selected
