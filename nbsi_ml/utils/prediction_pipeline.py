"""Prediction pipeline for fracture toughness."""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Any, List
import joblib


class PredictionPipeline:
    """
    End-to-end prediction pipeline for Nb-Si alloy fracture toughness.
    """
    
    def __init__(self, model_path: Optional[str] = None,
                 scaler_path: Optional[str] = None):
        """
        Initialize the PredictionPipeline.
        
        Args:
            model_path: Path to saved model
            scaler_path: Path to saved scaler
        """
        self.model = None
        self.scaler = None
        self.feature_names = None
        
        if model_path:
            self.load_model(model_path)
        if scaler_path:
            self.load_scaler(scaler_path)
    
    def load_model(self, model_path: str):
        """
        Load trained model from disk.
        
        Args:
            model_path: Path to the saved model
        """
        self.model = joblib.load(model_path)
        print(f"Model loaded from {model_path}")
    
    def load_scaler(self, scaler_path: str):
        """
        Load scaler from disk.
        
        Args:
            scaler_path: Path to the saved scaler
        """
        self.scaler = joblib.load(scaler_path)
        print(f"Scaler loaded from {scaler_path}")
    
    def set_feature_names(self, feature_names: List[str]):
        """
        Set feature names for the pipeline.
        
        Args:
            feature_names: List of feature names
        """
        self.feature_names = feature_names
    
    def preprocess_input(self, X: np.ndarray) -> np.ndarray:
        """
        Preprocess input features using the scaler.
        
        Args:
            X: Raw feature matrix
            
        Returns:
            Scaled feature matrix
        """
        if self.scaler is None:
            print("Warning: No scaler loaded. Returning raw features.")
            return X
        
        return self.scaler.transform(X)
    
    def predict(self, X: np.ndarray, preprocess: bool = True) -> np.ndarray:
        """
        Make predictions for fracture toughness.
        
        Args:
            X: Feature matrix (raw or preprocessed)
            preprocess: Whether to preprocess the input
            
        Returns:
            Predicted fracture toughness values
        """
        if self.model is None:
            raise ValueError("No model loaded. Load a model first.")
        
        if preprocess:
            X_processed = self.preprocess_input(X)
        else:
            X_processed = X
        
        predictions = self.model.predict(X_processed)
        return predictions
    
    def predict_from_dataframe(self, df: pd.DataFrame,
                              feature_columns: Optional[List[str]] = None,
                              preprocess: bool = True) -> np.ndarray:
        """
        Make predictions from a DataFrame.
        
        Args:
            df: Input DataFrame
            feature_columns: List of feature columns (if None, use self.feature_names)
            preprocess: Whether to preprocess the input
            
        Returns:
            Predicted fracture toughness values
        """
        if feature_columns is None:
            if self.feature_names is None:
                raise ValueError("Feature columns not specified and not set in pipeline.")
            feature_columns = self.feature_names
        
        X = df[feature_columns].values
        return self.predict(X, preprocess)
    
    def predict_single_sample(self, features: Dict[str, float],
                             preprocess: bool = True) -> float:
        """
        Make prediction for a single sample.
        
        Args:
            features: Dictionary of feature_name: value
            preprocess: Whether to preprocess the input
            
        Returns:
            Predicted fracture toughness value
        """
        if self.feature_names is None:
            raise ValueError("Feature names not set in pipeline.")
        
        # Create feature array in correct order
        X = np.array([[features[name] for name in self.feature_names]])
        prediction = self.predict(X, preprocess)
        
        return prediction[0]
    
    def predict_with_confidence(self, X: np.ndarray,
                               preprocess: bool = True,
                               n_estimators_threshold: int = 10) -> Dict:
        """
        Make predictions with confidence intervals (for ensemble models).
        
        Args:
            X: Feature matrix
            preprocess: Whether to preprocess the input
            n_estimators_threshold: Minimum number of estimators for confidence calculation
            
        Returns:
            Dictionary with predictions and confidence information
        """
        if self.model is None:
            raise ValueError("No model loaded.")
        
        if preprocess:
            X_processed = self.preprocess_input(X)
        else:
            X_processed = X
        
        # Get predictions
        predictions = self.model.predict(X_processed)
        
        # Try to get prediction intervals for ensemble models
        try:
            if hasattr(self.model, 'estimators_'):
                # Get predictions from all estimators
                estimator_predictions = np.array([
                    estimator.predict(X_processed) 
                    for estimator in self.model.estimators_
                ])
                
                # Calculate statistics
                pred_mean = estimator_predictions.mean(axis=0)
                pred_std = estimator_predictions.std(axis=0)
                pred_lower = np.percentile(estimator_predictions, 2.5, axis=0)
                pred_upper = np.percentile(estimator_predictions, 97.5, axis=0)
                
                return {
                    'predictions': predictions,
                    'mean': pred_mean,
                    'std': pred_std,
                    'lower_95': pred_lower,
                    'upper_95': pred_upper
                }
        except:
            pass
        
        # If confidence intervals not available, return just predictions
        return {
            'predictions': predictions,
            'mean': predictions,
            'std': None,
            'lower_95': None,
            'upper_95': None
        }
    
    def batch_predict(self, input_file: str, output_file: str,
                     feature_columns: Optional[List[str]] = None,
                     preprocess: bool = True):
        """
        Batch prediction from file to file.
        
        Args:
            input_file: Path to input CSV or Excel file
            output_file: Path to output CSV file
            feature_columns: List of feature columns
            preprocess: Whether to preprocess the input
        """
        # Load data
        if input_file.endswith('.csv'):
            df = pd.read_csv(input_file)
        elif input_file.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(input_file)
        else:
            raise ValueError("Unsupported file format. Use CSV or Excel.")
        
        # Make predictions
        predictions = self.predict_from_dataframe(df, feature_columns, preprocess)
        
        # Add predictions to DataFrame
        df['predicted_fracture_toughness'] = predictions
        
        # Save results
        df.to_csv(output_file, index=False)
        print(f"Predictions saved to {output_file}")
    
    def save_pipeline(self, model_path: str, scaler_path: str,
                     feature_names_path: str):
        """
        Save the complete pipeline.
        
        Args:
            model_path: Path to save the model
            scaler_path: Path to save the scaler
            feature_names_path: Path to save feature names
        """
        if self.model is not None:
            joblib.dump(self.model, model_path)
            print(f"Model saved to {model_path}")
        
        if self.scaler is not None:
            joblib.dump(self.scaler, scaler_path)
            print(f"Scaler saved to {scaler_path}")
        
        if self.feature_names is not None:
            joblib.dump(self.feature_names, feature_names_path)
            print(f"Feature names saved to {feature_names_path}")
    
    def load_pipeline(self, model_path: str, scaler_path: str,
                     feature_names_path: str):
        """
        Load the complete pipeline.
        
        Args:
            model_path: Path to the saved model
            scaler_path: Path to the saved scaler
            feature_names_path: Path to the saved feature names
        """
        self.load_model(model_path)
        self.load_scaler(scaler_path)
        self.feature_names = joblib.load(feature_names_path)
        print(f"Feature names loaded from {feature_names_path}")
        print(f"Pipeline loaded successfully with {len(self.feature_names)} features")
