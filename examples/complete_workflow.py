"""
Complete workflow example for NbSi alloy fracture toughness prediction.

This script demonstrates the entire ML pipeline including:
1. Data preprocessing
2. Feature engineering and selection
3. Model training
4. Model interpretation (SHAP, PDP)
5. Performance evaluation
6. Prediction pipeline
"""

import numpy as np
import pandas as pd
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nbsi_ml.preprocessing import DataPreprocessor
from nbsi_ml.feature_engineering import FeatureSelector
from nbsi_ml.models import ModelTrainer
from nbsi_ml.interpretability import ModelInterpreter
from nbsi_ml.evaluation import ModelEvaluator
from nbsi_ml.utils import PredictionPipeline, create_sample_data, ensure_directory


def main():
    """Run complete workflow."""
    
    print("=" * 80)
    print("NbSi ALLOY FRACTURE TOUGHNESS PREDICTION WORKFLOW")
    print("=" * 80)
    
    # Create output directories
    ensure_directory("./data")
    ensure_directory("./results")
    ensure_directory("./results/plots")
    ensure_directory("./results/models")
    ensure_directory("./results/interpretability")
    ensure_directory("./results/evaluation")
    
    # ============================================================================
    # 1. DATA PREPROCESSING
    # ============================================================================
    print("\n" + "=" * 80)
    print("STEP 1: DATA PREPROCESSING")
    print("=" * 80)
    
    # Create sample data for demonstration
    print("Creating sample dataset...")
    df = create_sample_data(n_samples=200, n_features=8)
    df.to_csv("./data/sample_alloy_data.csv", index=False)
    print(f"Sample data saved to ./data/sample_alloy_data.csv")
    print(f"Dataset shape: {df.shape}")
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(scaling_method="standard")
    
    # Preprocess data
    data = preprocessor.preprocess_pipeline(
        filepath="./data/sample_alloy_data.csv",
        target_column="fracture_toughness",
        test_size=0.2,
        random_state=42
    )
    
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    feature_names = data['feature_names']
    
    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    print(f"Features: {feature_names}")
    
    # ============================================================================
    # 2. FEATURE ENGINEERING AND SELECTION
    # ============================================================================
    print("\n" + "=" * 80)
    print("STEP 2: FEATURE ENGINEERING AND SELECTION")
    print("=" * 80)
    
    selector = FeatureSelector()
    
    # Calculate Pearson Correlation Coefficients
    print("\nCalculating Pearson Correlation Coefficients...")
    pcc_df = selector.calculate_pcc(X_train, y_train, feature_names)
    print("\nTop features by correlation:")
    print(pcc_df.head(10))
    
    # Plot correlation matrix
    selector.plot_correlation_matrix(
        X_train, y_train, feature_names,
        save_path="./results/plots/correlation_matrix.png"
    )
    
    # Plot feature importance
    selector.plot_feature_importance(
        pcc_df, top_n=len(feature_names),
        save_path="./results/plots/feature_importance_pcc.png"
    )
    
    # Compare RFE methods
    print("\nComparing RFE methods (Ridge, Huber, Random Forest)...")
    n_features_to_select = min(5, len(feature_names))
    rfe_comparison = selector.compare_rfe_methods(
        X_train, y_train, feature_names, n_features_to_select
    )
    print("\nRFE Comparison:")
    print(rfe_comparison)
    
    # ============================================================================
    # 3. MODEL TRAINING
    # ============================================================================
    print("\n" + "=" * 80)
    print("STEP 3: MODEL TRAINING")
    print("=" * 80)
    
    trainer = ModelTrainer()
    
    # Train all models
    print("\nTraining multiple models...")
    cv_results = trainer.train_all_models(X_train, y_train, cv=5)
    
    print("\nCross-validation results:")
    for model_name, results in cv_results.items():
        print(f"\n{model_name}:")
        print(f"  RMSE: {results['rmse_mean']:.4f} ± {results['mse_std']:.4f}")
        print(f"  MAE:  {results['mae_mean']:.4f} ± {results['mae_std']:.4f}")
        print(f"  R²:   {results['r2_mean']:.4f} ± {results['r2_std']:.4f}")
    
    # Save models
    for model_name in ['gradient_boosting', 'random_forest', 'bagging']:
        trainer.save_model(
            model_name, 
            f"./results/models/{model_name}.joblib"
        )
    
    # ============================================================================
    # 4. MODEL INTERPRETATION
    # ============================================================================
    print("\n" + "=" * 80)
    print("STEP 4: MODEL INTERPRETATION (SHAP & PDP)")
    print("=" * 80)
    
    # Use best model (typically Random Forest or Gradient Boosting)
    best_model = trainer.models['gradient_boosting']
    
    interpreter = ModelInterpreter()
    
    print("\nGenerating SHAP analysis...")
    interp_results = interpreter.comprehensive_interpretation(
        best_model, X_test, feature_names,
        model_type="tree",
        output_dir="./results/interpretability"
    )
    
    print("\nTop features by SHAP importance:")
    print(interp_results['feature_importance'].head(10))
    
    # ============================================================================
    # 5. PERFORMANCE EVALUATION
    # ============================================================================
    print("\n" + "=" * 80)
    print("STEP 5: PERFORMANCE EVALUATION")
    print("=" * 80)
    
    evaluator = ModelEvaluator()
    
    # Evaluate all models
    eval_results = evaluator.generate_evaluation_report(
        trainer.models, X_test, y_test,
        output_dir="./results/evaluation"
    )
    
    print("\nModel comparison on test set:")
    print(eval_results['results_df'])
    
    # ============================================================================
    # 6. PREDICTION PIPELINE
    # ============================================================================
    print("\n" + "=" * 80)
    print("STEP 6: PREDICTION PIPELINE")
    print("=" * 80)
    
    # Create prediction pipeline
    pipeline = PredictionPipeline()
    pipeline.model = best_model
    pipeline.scaler = data['scaler']
    pipeline.set_feature_names(feature_names)
    
    # Save complete pipeline
    pipeline.save_pipeline(
        model_path="./results/models/best_model.joblib",
        scaler_path="./results/models/scaler.joblib",
        feature_names_path="./results/models/feature_names.joblib"
    )
    
    # Make predictions on test set
    predictions = pipeline.predict(X_test, preprocess=False)
    
    print(f"\nSample predictions:")
    for i in range(min(5, len(predictions))):
        print(f"  Actual: {y_test[i]:.2f}, Predicted: {predictions[i]:.2f}")
    
    # Predict with confidence intervals
    conf_results = pipeline.predict_with_confidence(X_test, preprocess=False)
    if conf_results['std'] is not None:
        print("\nPrediction confidence intervals (first 5 samples):")
        for i in range(min(5, len(predictions))):
            print(f"  Prediction: {conf_results['mean'][i]:.2f} "
                  f"± {conf_results['std'][i]:.2f} "
                  f"[{conf_results['lower_95'][i]:.2f}, {conf_results['upper_95'][i]:.2f}]")
    
    # ============================================================================
    # SUMMARY
    # ============================================================================
    print("\n" + "=" * 80)
    print("WORKFLOW COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print("\nResults saved to:")
    print("  - Plots: ./results/plots/")
    print("  - Models: ./results/models/")
    print("  - Interpretability: ./results/interpretability/")
    print("  - Evaluation: ./results/evaluation/")
    print("\nBest model performance on test set:")
    best_result = eval_results['results_df'].iloc[0]
    print(f"  Model: {best_result['model_name']}")
    print(f"  R²: {best_result['r2']:.4f}")
    print(f"  RMSE: {best_result['rmse']:.4f}")
    print(f"  MAE: {best_result['mae']:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    # Run main workflow
    main()
