"""
Quick start example for NbSi alloy fracture toughness prediction.

Demonstrates basic usage of the framework with minimal code.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nbsi_ml.preprocessing import DataPreprocessor
from nbsi_ml.models import ModelTrainer
from nbsi_ml.evaluation import ModelEvaluator
from nbsi_ml.utils import create_sample_data, ensure_directory


def main():
    print("Quick Start: NbSi Alloy ML Framework\n")
    
    # Create sample data
    ensure_directory("./data")
    df = create_sample_data(n_samples=100, n_features=6)
    df.to_csv("./data/quick_start_data.csv", index=False)
    print(f"✓ Created sample data: {df.shape[0]} samples, {df.shape[1]-1} features")
    
    # Preprocess data
    preprocessor = DataPreprocessor()
    data = preprocessor.preprocess_pipeline(
        "./data/quick_start_data.csv",
        target_column="fracture_toughness"
    )
    print(f"✓ Preprocessed data: train={data['X_train'].shape}, test={data['X_test'].shape}")
    
    # Train model
    trainer = ModelTrainer()
    model = trainer.train_gradient_boosting(data['X_train'], data['y_train'])
    print("✓ Trained Gradient Boosting model")
    
    # Evaluate
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate_model(model, data['X_test'], data['y_test'], "GB")
    print(f"✓ Model Performance: R²={metrics['r2']:.4f}, RMSE={metrics['rmse']:.4f}")
    
    # Make predictions
    predictions = model.predict(data['X_test'][:5])
    print(f"\n✓ Sample predictions (first 5):")
    for i, (pred, actual) in enumerate(zip(predictions, data['y_test'][:5])):
        print(f"   #{i+1}: Predicted={pred:.2f}, Actual={actual:.2f}")
    
    print("\n✓ Quick start completed successfully!")


if __name__ == "__main__":
    main()
