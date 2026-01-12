"""Utility functions for the NbSi ML framework."""

import numpy as np
import pandas as pd
from typing import Dict, Any
import json
import os


def save_results_to_json(results: Dict, filepath: str):
    """
    Save results dictionary to JSON file.
    
    Args:
        results: Dictionary with results
        filepath: Path to save the JSON file
    """
    # Convert numpy types to Python types
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        return obj
    
    serializable_results = convert_to_serializable(results)
    
    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=4)
    
    print(f"Results saved to {filepath}")


def load_config(config_path: str) -> Dict:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def create_sample_data(n_samples: int = 100,
                      n_features: int = 10,
                      random_state: int = 42) -> pd.DataFrame:
    """
    Create sample alloy data for testing.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        random_state: Random seed
        
    Returns:
        Sample DataFrame
    """
    np.random.seed(random_state)
    
    # Generate composition features (percentages)
    data = {}
    elements = ['Nb', 'Si', 'Ti', 'Cr', 'Al', 'Mo', 'Hf', 'Ta', 'W', 'V']
    
    for i, element in enumerate(elements[:n_features]):
        data[f'{element}_content'] = np.random.uniform(0, 30, n_samples)
    
    # Generate target (fracture toughness) with some correlation to features
    fracture_toughness = (
        10 + 
        0.5 * data.get('Nb_content', np.zeros(n_samples)) +
        0.3 * data.get('Ti_content', np.zeros(n_samples)) -
        0.2 * data.get('Si_content', np.zeros(n_samples)) +
        np.random.normal(0, 2, n_samples)
    )
    
    # Ensure positive values and reasonable range
    fracture_toughness = np.clip(fracture_toughness, 5, 30)
    
    data['fracture_toughness'] = fracture_toughness
    
    return pd.DataFrame(data)


def print_summary_statistics(df: pd.DataFrame):
    """
    Print summary statistics for a DataFrame.
    
    Args:
        df: Input DataFrame
    """
    print("=" * 80)
    print("DATA SUMMARY STATISTICS")
    print("=" * 80)
    print(f"\nDataset shape: {df.shape}")
    print(f"Number of samples: {df.shape[0]}")
    print(f"Number of features: {df.shape[1]}")
    print("\nColumn names:")
    print(df.columns.tolist())
    print("\nMissing values:")
    print(df.isnull().sum())
    print("\nBasic statistics:")
    print(df.describe())
    print("=" * 80)


def ensure_directory(directory: str):
    """
    Ensure a directory exists, create if it doesn't.
    
    Args:
        directory: Path to directory
    """
    os.makedirs(directory, exist_ok=True)


def format_metrics_table(metrics: Dict[str, Any]) -> str:
    """
    Format metrics dictionary as a readable table string.
    
    Args:
        metrics: Dictionary of metrics
        
    Returns:
        Formatted string
    """
    lines = ["=" * 60, "MODEL PERFORMANCE METRICS", "=" * 60]
    
    for key, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"{key:30s}: {value:10.4f}")
        else:
            lines.append(f"{key:30s}: {value}")
    
    lines.append("=" * 60)
    return "\n".join(lines)
