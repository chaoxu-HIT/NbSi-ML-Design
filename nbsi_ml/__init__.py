"""
NbSi-ML-Design: Interpretable machine learning framework for designing 
Nb-Si based ultra-high temperature alloys with enhanced fracture toughness.
"""

__version__ = "0.1.0"
__author__ = "NbSi-ML-Design Team"

from . import preprocessing
from . import feature_engineering
from . import models
from . import interpretability
from . import evaluation
from . import utils

__all__ = [
    "preprocessing",
    "feature_engineering",
    "models",
    "interpretability",
    "evaluation",
    "utils",
]
