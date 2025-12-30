"""ML model training module with Gradient Boosting, Random Forest, and Bagging."""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Any
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    BaggingRegressor
)
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib


class ModelTrainer:
    """
    Train and manage multiple ML models for fracture toughness prediction.
    """
    
    def __init__(self):
        """Initialize the ModelTrainer."""
        self.models = {}
        self.best_params = {}
        self.cv_scores = {}
        
    def train_gradient_boosting(self, X_train: np.ndarray, y_train: np.ndarray,
                                n_estimators: int = 100,
                                learning_rate: float = 0.1,
                                max_depth: int = 3,
                                random_state: int = 42) -> GradientBoostingRegressor:
        """
        Train Gradient Boosting model.
        
        Args:
            X_train: Training features
            y_train: Training target
            n_estimators: Number of boosting stages
            learning_rate: Learning rate
            max_depth: Maximum depth of trees
            random_state: Random seed
            
        Returns:
            Trained model
        """
        model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state
        )
        model.fit(X_train, y_train)
        self.models['gradient_boosting'] = model
        return model
    
    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray,
                           n_estimators: int = 100,
                           max_depth: Optional[int] = None,
                           min_samples_split: int = 2,
                           random_state: int = 42) -> RandomForestRegressor:
        """
        Train Random Forest model.
        
        Args:
            X_train: Training features
            y_train: Training target
            n_estimators: Number of trees
            max_depth: Maximum depth of trees
            min_samples_split: Minimum samples required to split
            random_state: Random seed
            
        Returns:
            Trained model
        """
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        self.models['random_forest'] = model
        return model
    
    def train_bagging(self, X_train: np.ndarray, y_train: np.ndarray,
                     n_estimators: int = 100,
                     max_samples: float = 1.0,
                     random_state: int = 42) -> BaggingRegressor:
        """
        Train Bagging model.
        
        Args:
            X_train: Training features
            y_train: Training target
            n_estimators: Number of base estimators
            max_samples: Maximum samples to draw for each base estimator
            random_state: Random seed
            
        Returns:
            Trained model
        """
        model = BaggingRegressor(
            n_estimators=n_estimators,
            max_samples=max_samples,
            random_state=random_state,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        self.models['bagging'] = model
        return model
    
    def hyperparameter_tuning_gb(self, X_train: np.ndarray, y_train: np.ndarray,
                                 param_grid: Optional[Dict] = None,
                                 cv: int = 5) -> Dict:
        """
        Hyperparameter tuning for Gradient Boosting using GridSearchCV.
        
        Args:
            X_train: Training features
            y_train: Training target
            param_grid: Parameter grid for search
            cv: Number of cross-validation folds
            
        Returns:
            Dictionary with best parameters and model
        """
        if param_grid is None:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10]
            }
        
        gb = GradientBoostingRegressor(random_state=42)
        grid_search = GridSearchCV(
            gb, param_grid, cv=cv, scoring='neg_mean_squared_error',
            n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        self.best_params['gradient_boosting'] = grid_search.best_params_
        self.models['gradient_boosting_tuned'] = grid_search.best_estimator_
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': -grid_search.best_score_,
            'best_model': grid_search.best_estimator_
        }
    
    def hyperparameter_tuning_rf(self, X_train: np.ndarray, y_train: np.ndarray,
                                 param_grid: Optional[Dict] = None,
                                 cv: int = 5) -> Dict:
        """
        Hyperparameter tuning for Random Forest using GridSearchCV.
        
        Args:
            X_train: Training features
            y_train: Training target
            param_grid: Parameter grid for search
            cv: Number of cross-validation folds
            
        Returns:
            Dictionary with best parameters and model
        """
        if param_grid is None:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        
        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(
            rf, param_grid, cv=cv, scoring='neg_mean_squared_error',
            n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        self.best_params['random_forest'] = grid_search.best_params_
        self.models['random_forest_tuned'] = grid_search.best_estimator_
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': -grid_search.best_score_,
            'best_model': grid_search.best_estimator_
        }
    
    def cross_validate_model(self, model: Any, X: np.ndarray, y: np.ndarray,
                            cv: int = 5, model_name: str = "model") -> Dict:
        """
        Perform cross-validation on a model.
        
        Args:
            model: Model to cross-validate
            X: Feature matrix
            y: Target vector
            cv: Number of folds
            model_name: Name of the model
            
        Returns:
            Dictionary with cross-validation scores
        """
        # Calculate multiple metrics
        mse_scores = -cross_val_score(model, X, y, cv=cv, 
                                      scoring='neg_mean_squared_error')
        mae_scores = -cross_val_score(model, X, y, cv=cv, 
                                      scoring='neg_mean_absolute_error')
        r2_scores = cross_val_score(model, X, y, cv=cv, 
                                    scoring='r2')
        
        results = {
            'model_name': model_name,
            'mse_mean': np.mean(mse_scores),
            'mse_std': np.std(mse_scores),
            'mae_mean': np.mean(mae_scores),
            'mae_std': np.std(mae_scores),
            'r2_mean': np.mean(r2_scores),
            'r2_std': np.std(r2_scores),
            'rmse_mean': np.sqrt(np.mean(mse_scores))
        }
        
        self.cv_scores[model_name] = results
        return results
    
    def train_all_models(self, X_train: np.ndarray, y_train: np.ndarray,
                        cv: int = 5) -> Dict:
        """
        Train all available models and perform cross-validation.
        
        Args:
            X_train: Training features
            y_train: Training target
            cv: Number of cross-validation folds
            
        Returns:
            Dictionary with all trained models and CV scores
        """
        results = {}
        
        # Train Gradient Boosting
        print("Training Gradient Boosting...")
        gb_model = self.train_gradient_boosting(X_train, y_train)
        results['gradient_boosting'] = self.cross_validate_model(
            gb_model, X_train, y_train, cv, 'gradient_boosting'
        )
        
        # Train Random Forest
        print("Training Random Forest...")
        rf_model = self.train_random_forest(X_train, y_train)
        results['random_forest'] = self.cross_validate_model(
            rf_model, X_train, y_train, cv, 'random_forest'
        )
        
        # Train Bagging
        print("Training Bagging...")
        bag_model = self.train_bagging(X_train, y_train)
        results['bagging'] = self.cross_validate_model(
            bag_model, X_train, y_train, cv, 'bagging'
        )
        
        return results
    
    def predict(self, model_name: str, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using a trained model.
        
        Args:
            model_name: Name of the model to use
            X: Feature matrix
            
        Returns:
            Predictions
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Train it first.")
        
        return self.models[model_name].predict(X)
    
    def save_model(self, model_name: str, filepath: str):
        """
        Save a trained model to disk.
        
        Args:
            model_name: Name of the model to save
            filepath: Path to save the model
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found.")
        
        joblib.dump(self.models[model_name], filepath)
        print(f"Model '{model_name}' saved to {filepath}")
    
    def load_model(self, filepath: str, model_name: str):
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
            model_name: Name to assign to the loaded model
        """
        model = joblib.load(filepath)
        self.models[model_name] = model
        print(f"Model loaded from {filepath} as '{model_name}'")
        return model
