"""Model interpretability tools including SHAP analysis and PDP plots."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Tuple, Any, Dict
import shap
from sklearn.inspection import PartialDependenceDisplay


class ModelInterpreter:
    """
    Interpret ML models using SHAP and Partial Dependence Plots.
    """
    
    def __init__(self):
        """Initialize the ModelInterpreter."""
        self.explainer = None
        self.shap_values = None
        
    def create_shap_explainer(self, model: Any, X_background: np.ndarray,
                             model_type: str = "tree") -> Any:
        """
        Create SHAP explainer for the model.
        
        Args:
            model: Trained model
            X_background: Background data for SHAP
            model_type: Type of model ('tree', 'linear', 'kernel')
            
        Returns:
            SHAP explainer object
        """
        if model_type == "tree":
            self.explainer = shap.TreeExplainer(model)
        elif model_type == "linear":
            self.explainer = shap.LinearExplainer(model, X_background)
        elif model_type == "kernel":
            self.explainer = shap.KernelExplainer(model.predict, X_background)
        else:
            raise ValueError("model_type must be 'tree', 'linear', or 'kernel'")
        
        return self.explainer
    
    def calculate_shap_values(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate SHAP values for the dataset.
        
        Args:
            X: Feature matrix
            
        Returns:
            SHAP values
        """
        if self.explainer is None:
            raise ValueError("Create explainer first using create_shap_explainer()")
        
        self.shap_values = self.explainer.shap_values(X)
        return self.shap_values
    
    def plot_shap_summary(self, X: np.ndarray, feature_names: List[str],
                         max_display: int = 20,
                         save_path: Optional[str] = None):
        """
        Create SHAP summary plot (beeswarm plot).
        
        Args:
            X: Feature matrix
            feature_names: Names of features
            max_display: Maximum number of features to display
            save_path: Path to save the plot
        """
        if self.shap_values is None:
            self.calculate_shap_values(X)
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            self.shap_values, X, 
            feature_names=feature_names,
            max_display=max_display,
            show=False
        )
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_shap_bar(self, X: np.ndarray, feature_names: List[str],
                     max_display: int = 20,
                     save_path: Optional[str] = None):
        """
        Create SHAP bar plot showing mean absolute SHAP values.
        
        Args:
            X: Feature matrix
            feature_names: Names of features
            max_display: Maximum number of features to display
            save_path: Path to save the plot
        """
        if self.shap_values is None:
            self.calculate_shap_values(X)
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            self.shap_values, X,
            feature_names=feature_names,
            max_display=max_display,
            plot_type="bar",
            show=False
        )
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_shap_waterfall(self, X: np.ndarray, feature_names: List[str],
                           sample_idx: int = 0,
                           save_path: Optional[str] = None):
        """
        Create SHAP waterfall plot for a single prediction.
        
        Args:
            X: Feature matrix
            feature_names: Names of features
            sample_idx: Index of the sample to explain
            save_path: Path to save the plot
        """
        if self.shap_values is None:
            self.calculate_shap_values(X)
        
        # Create explanation object
        explanation = shap.Explanation(
            values=self.shap_values[sample_idx],
            base_values=self.explainer.expected_value,
            data=X[sample_idx],
            feature_names=feature_names
        )
        
        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(explanation, show=False)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_shap_dependence(self, X: np.ndarray, feature_names: List[str],
                            feature_idx: int,
                            interaction_idx: Optional[int] = None,
                            save_path: Optional[str] = None):
        """
        Create SHAP dependence plot for a specific feature.
        
        Args:
            X: Feature matrix
            feature_names: Names of features
            feature_idx: Index of the feature to plot
            interaction_idx: Index of interaction feature
            save_path: Path to save the plot
        """
        if self.shap_values is None:
            self.calculate_shap_values(X)
        
        plt.figure(figsize=(10, 6))
        
        if interaction_idx is not None:
            shap.dependence_plot(
                feature_idx, self.shap_values, X,
                feature_names=feature_names,
                interaction_index=interaction_idx,
                show=False
            )
        else:
            shap.dependence_plot(
                feature_idx, self.shap_values, X,
                feature_names=feature_names,
                show=False
            )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def get_feature_importance_shap(self, X: np.ndarray, 
                                   feature_names: List[str]) -> pd.DataFrame:
        """
        Get feature importance based on mean absolute SHAP values.
        
        Args:
            X: Feature matrix
            feature_names: Names of features
            
        Returns:
            DataFrame with feature importance
        """
        if self.shap_values is None:
            self.calculate_shap_values(X)
        
        # Calculate mean absolute SHAP values
        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': mean_abs_shap
        })
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        return importance_df
    
    def plot_partial_dependence(self, model: Any, X: np.ndarray,
                               feature_names: List[str],
                               features: List[int],
                               figsize: Tuple[int, int] = (12, 4),
                               save_path: Optional[str] = None):
        """
        Create Partial Dependence Plots (PDP).
        
        Args:
            model: Trained model
            X: Feature matrix
            feature_names: Names of features
            features: List of feature indices to plot
            figsize: Figure size
            save_path: Path to save the plot
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        display = PartialDependenceDisplay.from_estimator(
            model, X, features,
            feature_names=feature_names,
            ax=ax,
            n_jobs=-1
        )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_2d_partial_dependence(self, model: Any, X: np.ndarray,
                                   feature_names: List[str],
                                   features: Tuple[int, int],
                                   figsize: Tuple[int, int] = (8, 6),
                                   save_path: Optional[str] = None):
        """
        Create 2D Partial Dependence Plot for feature interactions.
        
        Args:
            model: Trained model
            X: Feature matrix
            feature_names: Names of features
            features: Tuple of two feature indices
            figsize: Figure size
            save_path: Path to save the plot
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        display = PartialDependenceDisplay.from_estimator(
            model, X, [features],
            feature_names=feature_names,
            ax=ax,
            n_jobs=-1
        )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def comprehensive_interpretation(self, model: Any, X: np.ndarray,
                                    feature_names: List[str],
                                    model_type: str = "tree",
                                    output_dir: str = "./interpretability_results") -> Dict:
        """
        Perform comprehensive model interpretation with all available tools.
        
        Args:
            model: Trained model
            X: Feature matrix
            feature_names: Names of features
            model_type: Type of model for SHAP
            output_dir: Directory to save plots
            
        Returns:
            Dictionary with interpretation results
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Create SHAP explainer and calculate values
        self.create_shap_explainer(model, X[:100], model_type)
        self.calculate_shap_values(X)
        
        # Generate SHAP plots
        self.plot_shap_summary(X, feature_names, 
                              save_path=f"{output_dir}/shap_summary.png")
        self.plot_shap_bar(X, feature_names,
                          save_path=f"{output_dir}/shap_bar.png")
        
        # Get feature importance
        importance_df = self.get_feature_importance_shap(X, feature_names)
        importance_df.to_csv(f"{output_dir}/shap_feature_importance.csv", index=False)
        
        # Create PDP for top features
        top_features = list(range(min(5, len(feature_names))))
        self.plot_partial_dependence(model, X, feature_names, top_features,
                                    save_path=f"{output_dir}/partial_dependence.png")
        
        return {
            'feature_importance': importance_df,
            'shap_values': self.shap_values,
            'output_directory': output_dir
        }
