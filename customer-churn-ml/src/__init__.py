"""
Customer Churn ML project package initialization.

This module makes the src directory a Python package and provides
convenient imports for the main functionality.
"""

from .config import *
from .preprocess import load_raw_data, clean_data, prepare_features_target
from .predict import ChurnPredictor, predict_churn, batch_predict
from .evaluate import calculate_classification_metrics, generate_model_report
from .utils import setup_logging, get_data_quality_report

__version__ = "1.0.0"
__author__ = "ML Engineering Team"
__description__ = "Customer Churn Prediction ML Pipeline"

# Configure logging by default
setup_logging()