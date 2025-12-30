"""Data preprocessing module for alloy composition and mechanical properties."""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split


class DataPreprocessor:
    """
    Preprocessor for alloy composition and mechanical property data.
    
    Handles data loading, cleaning, normalization, and train-test splitting.
    """
    
    def __init__(self, scaling_method: str = "standard"):
        """
        Initialize the DataPreprocessor.
        
        Args:
            scaling_method: Scaling method to use ('standard' or 'minmax')
        """
        self.scaling_method = scaling_method
        self.scaler = None
        self.feature_columns = None
        self.target_column = None
        
    def load_data(self, filepath: str, target_column: str = "fracture_toughness") -> pd.DataFrame:
        """
        Load data from CSV or Excel file.
        
        Args:
            filepath: Path to the data file
            target_column: Name of the target variable column
            
        Returns:
            Loaded DataFrame
        """
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filepath.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(filepath)
        else:
            raise ValueError("Unsupported file format. Use CSV or Excel.")
        
        self.target_column = target_column
        return df
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = "mean") -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df: Input DataFrame
            strategy: Strategy for handling missing values ('mean', 'median', 'drop')
            
        Returns:
            DataFrame with handled missing values
        """
        df_copy = df.copy()
        
        if strategy == "drop":
            df_copy = df_copy.dropna()
        elif strategy == "mean":
            numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
            df_copy[numeric_cols] = df_copy[numeric_cols].fillna(df_copy[numeric_cols].mean())
        elif strategy == "median":
            numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
            df_copy[numeric_cols] = df_copy[numeric_cols].fillna(df_copy[numeric_cols].median())
        else:
            raise ValueError("Strategy must be 'mean', 'median', or 'drop'")
        
        return df_copy
    
    def normalize_data(self, X: np.ndarray, fit: bool = True) -> np.ndarray:
        """
        Normalize/standardize features.
        
        Args:
            X: Feature matrix
            fit: Whether to fit the scaler (True for training data)
            
        Returns:
            Normalized feature matrix
        """
        if self.scaler is None or fit:
            if self.scaling_method == "standard":
                self.scaler = StandardScaler()
            elif self.scaling_method == "minmax":
                self.scaler = MinMaxScaler()
            else:
                raise ValueError("Scaling method must be 'standard' or 'minmax'")
            
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        return X_scaled
    
    def prepare_features(self, df: pd.DataFrame, 
                        feature_columns: Optional[List[str]] = None,
                        exclude_columns: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and target from DataFrame.
        
        Args:
            df: Input DataFrame
            feature_columns: List of feature column names (if None, use all except target)
            exclude_columns: List of columns to exclude
            
        Returns:
            Tuple of (X, y) where X is feature matrix and y is target vector
        """
        df_copy = df.copy()
        
        if feature_columns is None:
            # Use all columns except target and excluded columns
            feature_columns = [col for col in df_copy.columns if col != self.target_column]
            if exclude_columns:
                feature_columns = [col for col in feature_columns if col not in exclude_columns]
        
        self.feature_columns = feature_columns
        
        X = df_copy[feature_columns].values
        y = df_copy[self.target_column].values
        
        return X, y
    
    def split_data(self, X: np.ndarray, y: np.ndarray, 
                   test_size: float = 0.2, 
                   random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into training and testing sets.
        
        Args:
            X: Feature matrix
            y: Target vector
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    def preprocess_pipeline(self, filepath: str, 
                           target_column: str = "fracture_toughness",
                           test_size: float = 0.2,
                           missing_strategy: str = "mean",
                           feature_columns: Optional[List[str]] = None,
                           exclude_columns: Optional[List[str]] = None,
                           random_state: int = 42) -> dict:
        """
        Complete preprocessing pipeline from file to train-test split.
        
        Args:
            filepath: Path to the data file
            target_column: Name of the target variable
            test_size: Proportion of data for testing
            missing_strategy: Strategy for handling missing values
            feature_columns: List of feature columns to use
            exclude_columns: List of columns to exclude
            random_state: Random seed
            
        Returns:
            Dictionary containing preprocessed data and metadata
        """
        # Load data
        df = self.load_data(filepath, target_column)
        
        # Handle missing values
        df = self.handle_missing_values(df, missing_strategy)
        
        # Prepare features
        X, y = self.prepare_features(df, feature_columns, exclude_columns)
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y, test_size, random_state)
        
        # Normalize features
        X_train_scaled = self.normalize_data(X_train, fit=True)
        X_test_scaled = self.normalize_data(X_test, fit=False)
        
        return {
            "X_train": X_train_scaled,
            "X_test": X_test_scaled,
            "y_train": y_train,
            "y_test": y_test,
            "X_train_raw": X_train,
            "X_test_raw": X_test,
            "feature_names": self.feature_columns,
            "scaler": self.scaler,
            "dataframe": df
        }
