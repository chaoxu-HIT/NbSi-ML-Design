"""Basic tests for the NbSi ML framework."""

import unittest
import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nbsi_ml.preprocessing import DataPreprocessor
from nbsi_ml.feature_engineering import FeatureSelector
from nbsi_ml.models import ModelTrainer
from nbsi_ml.interpretability import ModelInterpreter
from nbsi_ml.evaluation import ModelEvaluator
from nbsi_ml.utils import PredictionPipeline, create_sample_data


class TestDataPreprocessor(unittest.TestCase):
    """Test DataPreprocessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.preprocessor = DataPreprocessor()
        self.df = create_sample_data(n_samples=50, n_features=5)
    
    def test_handle_missing_values(self):
        """Test missing value handling."""
        df_with_na = self.df.copy()
        df_with_na.iloc[0, 0] = np.nan
        result = self.preprocessor.handle_missing_values(df_with_na, strategy="mean")
        self.assertEqual(result.isnull().sum().sum(), 0)
    
    def test_prepare_features(self):
        """Test feature preparation."""
        self.preprocessor.target_column = "fracture_toughness"
        X, y = self.preprocessor.prepare_features(self.df)
        self.assertEqual(X.shape[0], len(self.df))
        self.assertEqual(len(y), len(self.df))
    
    def test_normalize_data(self):
        """Test data normalization."""
        X = np.random.randn(10, 3)
        X_scaled = self.preprocessor.normalize_data(X, fit=True)
        self.assertAlmostEqual(X_scaled.mean(), 0, places=5)


class TestFeatureSelector(unittest.TestCase):
    """Test FeatureSelector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.selector = FeatureSelector()
        self.X = np.random.randn(50, 5)
        self.y = np.random.randn(50)
        self.feature_names = [f'feature_{i}' for i in range(5)]
    
    def test_calculate_pcc(self):
        """Test PCC calculation."""
        pcc_df = self.selector.calculate_pcc(self.X, self.y, self.feature_names)
        self.assertEqual(len(pcc_df), 5)
        self.assertIn('correlation', pcc_df.columns)
        self.assertIn('p_value', pcc_df.columns)
    
    def test_rfe_ridge(self):
        """Test RFE with Ridge."""
        results = self.selector.rfe_ridge(self.X, self.y, self.feature_names, n_features_to_select=3)
        self.assertEqual(len(results['selected_features']), 3)
        self.assertEqual(results['model'], 'Ridge')


class TestModelTrainer(unittest.TestCase):
    """Test ModelTrainer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.trainer = ModelTrainer()
        self.X_train = np.random.randn(50, 5)
        self.y_train = np.random.randn(50)
    
    def test_train_gradient_boosting(self):
        """Test Gradient Boosting training."""
        model = self.trainer.train_gradient_boosting(self.X_train, self.y_train)
        self.assertIsNotNone(model)
        predictions = model.predict(self.X_train[:5])
        self.assertEqual(len(predictions), 5)
    
    def test_train_random_forest(self):
        """Test Random Forest training."""
        model = self.trainer.train_random_forest(self.X_train, self.y_train)
        self.assertIsNotNone(model)
        predictions = model.predict(self.X_train[:5])
        self.assertEqual(len(predictions), 5)
    
    def test_cross_validate_model(self):
        """Test cross-validation."""
        model = self.trainer.train_gradient_boosting(self.X_train, self.y_train)
        results = self.trainer.cross_validate_model(model, self.X_train, self.y_train, cv=3)
        self.assertIn('mse_mean', results)
        self.assertIn('r2_mean', results)


class TestModelEvaluator(unittest.TestCase):
    """Test ModelEvaluator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.evaluator = ModelEvaluator()
        self.y_true = np.random.randn(20)
        self.y_pred = np.random.randn(20)
    
    def test_calculate_metrics(self):
        """Test metrics calculation."""
        metrics = self.evaluator.calculate_metrics(self.y_true, self.y_pred)
        self.assertIn('mse', metrics)
        self.assertIn('rmse', metrics)
        self.assertIn('mae', metrics)
        self.assertIn('r2', metrics)


class TestPredictionPipeline(unittest.TestCase):
    """Test PredictionPipeline class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.pipeline = PredictionPipeline()
        self.X = np.random.randn(10, 5)
        self.feature_names = [f'feature_{i}' for i in range(5)]
    
    def test_predict_without_model(self):
        """Test prediction without model raises error."""
        with self.assertRaises(ValueError):
            self.pipeline.predict(self.X)
    
    def test_set_feature_names(self):
        """Test setting feature names."""
        self.pipeline.set_feature_names(self.feature_names)
        self.assertEqual(len(self.pipeline.feature_names), 5)


class TestIntegration(unittest.TestCase):
    """Integration tests for complete workflow."""
    
    def test_complete_workflow(self):
        """Test complete workflow from data to prediction."""
        # Create sample data
        df = create_sample_data(n_samples=80, n_features=5)
        
        # Preprocess
        preprocessor = DataPreprocessor()
        preprocessor.target_column = "fracture_toughness"
        X, y = preprocessor.prepare_features(df)
        X_train, X_test, y_train, y_test = preprocessor.split_data(X, y, test_size=0.2)
        X_train_scaled = preprocessor.normalize_data(X_train, fit=True)
        X_test_scaled = preprocessor.normalize_data(X_test, fit=False)
        
        # Train model
        trainer = ModelTrainer()
        model = trainer.train_gradient_boosting(X_train_scaled, y_train, n_estimators=10)
        
        # Evaluate
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate_model(model, X_test_scaled, y_test)
        
        # Check results
        self.assertIsNotNone(metrics['r2'])
        self.assertIsNotNone(metrics['rmse'])
        
        # Make predictions
        predictions = model.predict(X_test_scaled)
        self.assertEqual(len(predictions), len(y_test))


if __name__ == '__main__':
    unittest.main()
