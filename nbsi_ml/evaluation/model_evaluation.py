"""Performance evaluation module for ML models."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error
)


class ModelEvaluator:
    """
    Evaluate and compare ML model performance.
    """
    
    def __init__(self):
        """Initialize the ModelEvaluator."""
        self.evaluation_results = {}
        
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                         model_name: str = "model") -> Dict:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            model_name: Name of the model
            
        Returns:
            Dictionary with metrics
        """
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Calculate MAPE if no zeros in y_true
        try:
            mape = mean_absolute_percentage_error(y_true, y_pred)
        except (ZeroDivisionError, ValueError):
            mape = None
        
        metrics = {
            'model_name': model_name,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape
        }
        
        self.evaluation_results[model_name] = metrics
        return metrics
    
    def evaluate_model(self, model: Any, X_test: np.ndarray, y_test: np.ndarray,
                      model_name: str = "model") -> Dict:
        """
        Evaluate a model on test data.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: True test values
            model_name: Name of the model
            
        Returns:
            Dictionary with metrics
        """
        y_pred = model.predict(X_test)
        return self.calculate_metrics(y_test, y_pred, model_name)
    
    def evaluate_multiple_models(self, models: Dict[str, Any],
                                X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
        """
        Evaluate multiple models and compare results.
        
        Args:
            models: Dictionary of model_name: model
            X_test: Test features
            y_test: True test values
            
        Returns:
            DataFrame with comparison results
        """
        results = []
        
        for model_name, model in models.items():
            metrics = self.evaluate_model(model, X_test, y_test, model_name)
            results.append(metrics)
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('r2', ascending=False)
        
        return results_df
    
    def plot_predictions_vs_actual(self, y_true: np.ndarray, y_pred: np.ndarray,
                                   model_name: str = "model",
                                   figsize: Tuple[int, int] = (8, 8),
                                   save_path: Optional[str] = None):
        """
        Plot predicted vs actual values.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name of the model
            figsize: Figure size
            save_path: Path to save the plot
        """
        plt.figure(figsize=figsize)
        
        # Scatter plot
        plt.scatter(y_true, y_pred, alpha=0.6, edgecolors='k', linewidths=0.5)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        # Calculate R²
        r2 = r2_score(y_true, y_pred)
        
        plt.xlabel('Actual Fracture Toughness (MPa·m^1/2)', fontsize=12)
        plt.ylabel('Predicted Fracture Toughness (MPa·m^1/2)', fontsize=12)
        plt.title(f'{model_name} - Predictions vs Actual\nR² = {r2:.4f}', 
                 fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_residuals(self, y_true: np.ndarray, y_pred: np.ndarray,
                      model_name: str = "model",
                      figsize: Tuple[int, int] = (12, 5),
                      save_path: Optional[str] = None):
        """
        Plot residual analysis.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name of the model
            figsize: Figure size
            save_path: Path to save the plot
        """
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Residuals vs Predicted
        axes[0].scatter(y_pred, residuals, alpha=0.6, edgecolors='k', linewidths=0.5)
        axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0].set_xlabel('Predicted Values', fontsize=11)
        axes[0].set_ylabel('Residuals', fontsize=11)
        axes[0].set_title('Residuals vs Predicted', fontsize=12, fontweight='bold')
        axes[0].grid(alpha=0.3)
        
        # Residual histogram
        axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        axes[1].axvline(x=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('Residuals', fontsize=11)
        axes[1].set_ylabel('Frequency', fontsize=11)
        axes[1].set_title('Residual Distribution', fontsize=12, fontweight='bold')
        axes[1].grid(alpha=0.3)
        
        plt.suptitle(f'{model_name} - Residual Analysis', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_model_comparison(self, results_df: pd.DataFrame,
                             metric: str = 'r2',
                             figsize: Tuple[int, int] = (10, 6),
                             save_path: Optional[str] = None):
        """
        Plot comparison of models by a specific metric.
        
        Args:
            results_df: DataFrame with model comparison results
            metric: Metric to compare ('r2', 'rmse', 'mae', 'mse')
            figsize: Figure size
            save_path: Path to save the plot
        """
        plt.figure(figsize=figsize)
        
        # Determine if higher is better
        higher_is_better = metric in ['r2']
        
        if higher_is_better:
            sorted_df = results_df.sort_values(metric, ascending=True)
            color = 'green'
        else:
            sorted_df = results_df.sort_values(metric, ascending=False)
            color = 'red'
        
        plt.barh(range(len(sorted_df)), sorted_df[metric], color=color, alpha=0.7, edgecolor='black')
        plt.yticks(range(len(sorted_df)), sorted_df['model_name'])
        plt.xlabel(metric.upper(), fontsize=12)
        plt.ylabel('Model', fontsize=12)
        plt.title(f'Model Comparison by {metric.upper()}', fontsize=14, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_learning_curve(self, model: Any, X: np.ndarray, y: np.ndarray,
                           train_sizes: np.ndarray = np.linspace(0.1, 1.0, 10),
                           cv: int = 5,
                           figsize: Tuple[int, int] = (10, 6),
                           save_path: Optional[str] = None):
        """
        Plot learning curve to diagnose bias/variance.
        
        Args:
            model: Model to evaluate
            X: Feature matrix
            y: Target vector
            train_sizes: Array of training set sizes
            cv: Number of CV folds
            figsize: Figure size
            save_path: Path to save the plot
        """
        from sklearn.model_selection import learning_curve
        
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, train_sizes=train_sizes, cv=cv,
            scoring='neg_mean_squared_error', n_jobs=-1
        )
        
        # Convert to RMSE
        train_scores_mean = np.sqrt(-train_scores.mean(axis=1))
        train_scores_std = np.sqrt(train_scores.std(axis=1))
        val_scores_mean = np.sqrt(-val_scores.mean(axis=1))
        val_scores_std = np.sqrt(val_scores.std(axis=1))
        
        plt.figure(figsize=figsize)
        plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training RMSE')
        plt.fill_between(train_sizes, 
                        train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, 
                        alpha=0.1, color='r')
        
        plt.plot(train_sizes, val_scores_mean, 'o-', color='g', label='Validation RMSE')
        plt.fill_between(train_sizes,
                        val_scores_mean - val_scores_std,
                        val_scores_mean + val_scores_std,
                        alpha=0.1, color='g')
        
        plt.xlabel('Training Set Size', fontsize=12)
        plt.ylabel('RMSE', fontsize=12)
        plt.title('Learning Curve', fontsize=14, fontweight='bold')
        plt.legend(loc='best')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_evaluation_report(self, models: Dict[str, Any],
                                  X_test: np.ndarray, y_test: np.ndarray,
                                  output_dir: str = "./evaluation_results") -> Dict:
        """
        Generate comprehensive evaluation report for all models.
        
        Args:
            models: Dictionary of model_name: model
            X_test: Test features
            y_test: True test values
            output_dir: Directory to save results
            
        Returns:
            Dictionary with evaluation results
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Evaluate all models
        results_df = self.evaluate_multiple_models(models, X_test, y_test)
        results_df.to_csv(f"{output_dir}/model_comparison.csv", index=False)
        
        # Generate plots for each model
        for model_name, model in models.items():
            y_pred = model.predict(X_test)
            
            # Predictions vs Actual
            self.plot_predictions_vs_actual(
                y_test, y_pred, model_name,
                save_path=f"{output_dir}/{model_name}_predictions.png"
            )
            
            # Residuals
            self.plot_residuals(
                y_test, y_pred, model_name,
                save_path=f"{output_dir}/{model_name}_residuals.png"
            )
        
        # Model comparison plot
        self.plot_model_comparison(
            results_df, metric='r2',
            save_path=f"{output_dir}/model_comparison_r2.png"
        )
        
        self.plot_model_comparison(
            results_df, metric='rmse',
            save_path=f"{output_dir}/model_comparison_rmse.png"
        )
        
        return {
            'results_df': results_df,
            'output_directory': output_dir
        }
