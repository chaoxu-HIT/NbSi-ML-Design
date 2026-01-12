# NbSi-ML-Design

Interpretable machine learning framework for designing Nb-Si based ultra-high temperature alloys with enhanced fracture toughness (>20 MPa·m^1/2).

## Overview

This framework provides a comprehensive suite of tools for machine learning-based design and analysis of Nb-Si alloys, with a focus on predicting and optimizing fracture toughness. The framework includes:

- **Data Preprocessing**: Handle alloy composition data and mechanical properties
- **Feature Engineering**: PCC analysis and RFE with Ridge, Huber, and Random Forest
- **ML Models**: Gradient Boosting, Random Forest, and Bagging regressors
- **Interpretability**: SHAP analysis and Partial Dependence Plots
- **Evaluation**: Comprehensive performance metrics and visualizations
- **Prediction Pipeline**: Production-ready prediction system

## Features

### 1. Data Preprocessing Module (`nbsi_ml.preprocessing`)
- Load data from CSV/Excel files
- Handle missing values (mean, median, drop strategies)
- Normalize/standardize features (StandardScaler, MinMaxScaler)
- Train-test splitting
- Complete preprocessing pipeline

### 2. Feature Engineering and Screening (`nbsi_ml.feature_engineering`)
- **Pearson Correlation Coefficient (PCC)** analysis
- **Recursive Feature Elimination (RFE)** with:
  - Ridge regression
  - Huber regression
  - Random Forest
- Correlation matrix visualization
- Feature importance plotting
- Multi-method comparison

### 3. ML Model Training (`nbsi_ml.models`)
- **Gradient Boosting** regressor
- **Random Forest** regressor
- **Bagging** regressor
- Hyperparameter tuning with GridSearchCV
- Cross-validation
- Model persistence (save/load)

### 4. Model Interpretability (`nbsi_ml.interpretability`)
- **SHAP Analysis**:
  - Summary plots (beeswarm)
  - Bar plots
  - Waterfall plots
  - Dependence plots
  - Feature importance
- **Partial Dependence Plots (PDP)**:
  - 1D PDP
  - 2D PDP for interactions

### 5. Performance Evaluation (`nbsi_ml.evaluation`)
- Comprehensive metrics (RMSE, MAE, R², MAPE)
- Predictions vs Actual plots
- Residual analysis
- Model comparison
- Learning curves
- Evaluation reports

### 6. Prediction Pipeline (`nbsi_ml.utils`)
- End-to-end prediction workflow
- Single sample prediction
- Batch prediction from files
- Confidence intervals (for ensemble models)
- Pipeline persistence

## Installation

1. Clone the repository:
```bash
git clone https://github.com/chaoxu-HIT/NbSi-ML-Design.git
cd NbSi-ML-Design
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

```python
from nbsi_ml.preprocessing import DataPreprocessor
from nbsi_ml.models import ModelTrainer
from nbsi_ml.evaluation import ModelEvaluator

# Preprocess data
preprocessor = DataPreprocessor()
data = preprocessor.preprocess_pipeline(
    "path/to/your/data.csv",
    target_column="fracture_toughness"
)

# Train model
trainer = ModelTrainer()
model = trainer.train_gradient_boosting(data['X_train'], data['y_train'])

# Evaluate
evaluator = ModelEvaluator()
metrics = evaluator.evaluate_model(model, data['X_test'], data['y_test'])
print(f"R²: {metrics['r2']:.4f}, RMSE: {metrics['rmse']:.4f}")

# Make predictions
predictions = model.predict(data['X_test'])
```

See `examples/quick_start.py` for a complete minimal example.

## Usage Examples

### Complete Workflow

Run the complete workflow example:
```bash
python examples/complete_workflow.py
```

This demonstrates:
1. Data loading and preprocessing
2. Feature selection with PCC and RFE
3. Training multiple models with cross-validation
4. SHAP analysis and PDP plots
5. Model evaluation and comparison
6. Setting up prediction pipeline

### Custom Workflow

```python
from nbsi_ml.preprocessing import DataPreprocessor
from nbsi_ml.feature_engineering import FeatureSelector
from nbsi_ml.models import ModelTrainer
from nbsi_ml.interpretability import ModelInterpreter
from nbsi_ml.evaluation import ModelEvaluator
from nbsi_ml.utils import PredictionPipeline

# 1. Preprocess
preprocessor = DataPreprocessor(scaling_method="standard")
data = preprocessor.preprocess_pipeline("data.csv", target_column="fracture_toughness")

# 2. Feature Selection
selector = FeatureSelector()
pcc_df = selector.calculate_pcc(data['X_train'], data['y_train'], data['feature_names'])
rfe_results = selector.rfe_ridge(data['X_train'], data['y_train'], data['feature_names'])

# 3. Train Models
trainer = ModelTrainer()
cv_results = trainer.train_all_models(data['X_train'], data['y_train'])

# 4. Interpret
interpreter = ModelInterpreter()
interp_results = interpreter.comprehensive_interpretation(
    trainer.models['gradient_boosting'],
    data['X_test'],
    data['feature_names']
)

# 5. Evaluate
evaluator = ModelEvaluator()
eval_results = evaluator.generate_evaluation_report(
    trainer.models, data['X_test'], data['y_test']
)

# 6. Create Prediction Pipeline
pipeline = PredictionPipeline()
pipeline.model = trainer.models['gradient_boosting']
pipeline.scaler = data['scaler']
pipeline.set_feature_names(data['feature_names'])
pipeline.save_pipeline("model.joblib", "scaler.joblib", "features.joblib")
```

## Data Format

Your input data should be in CSV or Excel format with:
- One column for the target variable (e.g., "fracture_toughness")
- Multiple columns for features (e.g., alloy composition, processing parameters)

Example:
```
Nb_content,Si_content,Ti_content,Cr_content,fracture_toughness
45.2,15.3,8.5,2.1,18.5
42.8,16.1,9.2,2.5,19.2
...
```

## Project Structure

```
NbSi-ML-Design/
├── nbsi_ml/                    # Main package
│   ├── preprocessing/          # Data preprocessing
│   ├── feature_engineering/    # Feature selection
│   ├── models/                 # ML models
│   ├── interpretability/       # SHAP and PDP
│   ├── evaluation/             # Performance evaluation
│   └── utils/                  # Utilities and pipeline
├── examples/                   # Usage examples
│   ├── complete_workflow.py   # Complete workflow demo
│   └── quick_start.py         # Quick start example
├── data/                       # Data directory
├── tests/                      # Unit tests
├── requirements.txt            # Dependencies
└── README.md                   # Documentation
```

## Requirements

- Python >= 3.7
- numpy >= 1.21.0
- pandas >= 1.3.0
- scikit-learn >= 1.0.0
- shap >= 0.41.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- xgboost >= 1.5.0

See `requirements.txt` for complete list.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{nbsi_ml_design,
  title = {NbSi-ML-Design: Interpretable Machine Learning for Nb-Si Alloy Design},
  author = {NbSi-ML-Design Team},
  year = {2024},
  url = {https://github.com/chaoxu-HIT/NbSi-ML-Design}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or issues, please open an issue on GitHub.
