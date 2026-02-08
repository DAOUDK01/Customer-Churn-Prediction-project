"""
Configuration file for Customer Churn ML project.

Contains paths, model parameters, and other constants used across the project.
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data paths
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed"

# Model paths
MODELS_PATH = PROJECT_ROOT / "models"
MODEL_PIPELINE_PATH = MODELS_PATH / "churn_prediction_pipeline.joblib"

# Reports paths
REPORTS_PATH = PROJECT_ROOT / "reports"

# Feature columns (update based on your actual dataset)
FEATURE_COLUMNS = [
    'tenure', 'MonthlyCharges', 'TotalCharges', 'Contract_length',
    'PaymentMethod', 'InternetService', 'TechSupport', 'OnlineSecurity'
]

# Target column
TARGET_COLUMN = 'Churn'

# Model parameters
MODEL_CONFIG = {
    'random_state': 42,
    'test_size': 0.2,
    'cv_folds': 5
}

# Data preprocessing parameters
PREPROCESSING_CONFIG = {
    'missing_value_threshold': 0.1,  # Drop columns with >10% missing values
    'categorical_encoding': 'onehot',
    'numerical_scaling': 'standard'
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
}