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

# Model Registry
MODEL_REGISTRY = MODELS_PATH / "simple_churn_model.joblib"

# API Configuration
API_CONFIG = {
    'host': '0.0.0.0',
    'port': 8000,
    'title': 'Customer Churn Prediction API',
    'version': '2.0.0',
    'max_batch_size': 1000,
    'rate_limit': 1000,  # requests per minute
    'auth_token': 'demo-token-2026'
}

# Monitoring Configuration
MONITORING_CONFIG = {
    'drift_threshold': 0.1,
    'performance_threshold': 0.05,
    'alert_threshold': 0.15,
    'monitoring_window_days': 7,
    'baseline_update_frequency_days': 30
}

# Batch Processing Configuration
BATCH_CONFIG = {
    'batch_size': 1000,
    'max_workers': 4,
    'chunk_size': 10000,
    'max_memory_mb': 2048,
    'timeout_seconds': 3600
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