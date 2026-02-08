"""
Data preprocessing utilities for Customer Churn ML project.

Contains functions for loading, cleaning, and transforming data.
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Optional, List
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from .config import RAW_DATA_PATH, PROCESSED_DATA_PATH, FEATURE_COLUMNS, TARGET_COLUMN

logger = logging.getLogger(__name__)


def load_raw_data(file_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load raw customer data from CSV file.
    
    Args:
        file_path: Path to CSV file. If None, uses default raw data path.
        
    Returns:
        DataFrame with raw customer data.
        
    Raises:
        FileNotFoundError: If the data file doesn't exist.
    """
    if file_path is None:
        file_path = RAW_DATA_PATH / "customer_data.csv"
    
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded raw data with shape: {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"Data file not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw data by handling missing values and data types.
    
    Args:
        df: Raw DataFrame to clean.
        
    Returns:
        Cleaned DataFrame.
    """
    df_clean = df.copy()
    
    # Convert TotalCharges to numeric (handle spaces and empty strings)
    if 'TotalCharges' in df_clean.columns:
        df_clean['TotalCharges'] = pd.to_numeric(
            df_clean['TotalCharges'].replace(' ', np.nan), 
            errors='coerce'
        )
    
    # Handle missing values
    numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
    categorical_columns = df_clean.select_dtypes(include=['object']).columns
    
    # Fill numeric missing values with median
    for col in numeric_columns:
        if df_clean[col].isnull().any():
            median_val = df_clean[col].median()
            df_clean[col].fillna(median_val, inplace=True)
            logger.info(f"Filled {col} missing values with median: {median_val}")
    
    # Fill categorical missing values with mode
    for col in categorical_columns:
        if df_clean[col].isnull().any():
            mode_val = df_clean[col].mode().iloc[0] if not df_clean[col].mode().empty else 'Unknown'
            df_clean[col].fillna(mode_val, inplace=True)
            logger.info(f"Filled {col} missing values with mode: {mode_val}")
    
    logger.info(f"Data cleaning complete. Shape: {df_clean.shape}")
    return df_clean


def encode_target(y: pd.Series) -> Tuple[np.ndarray, LabelEncoder]:
    """
    Encode target variable for binary classification.
    
    Args:
        y: Target series to encode.
        
    Returns:
        Tuple of (encoded_target, label_encoder)
    """
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    logger.info(f"Target encoded. Classes: {le.classes_}")
    return y_encoded, le


def create_preprocessor() -> ColumnTransformer:
    """
    Create preprocessing pipeline for features.
    
    Returns:
        ColumnTransformer for preprocessing features.
    """
    # Identify numeric and categorical columns
    # Note: This is a template - adjust based on your actual data
    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_features = [
        'Contract', 'PaymentMethod', 'InternetService', 
        'TechSupport', 'OnlineSecurity'
    ]
    
    # Create preprocessing pipelines
    numeric_pipeline = Pipeline([
        ('scaler', StandardScaler())
    ])
    
    categorical_pipeline = Pipeline([
        ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer([
        ('num', numeric_pipeline, numeric_features),
        ('cat', categorical_pipeline, categorical_features)
    ])
    
    logger.info("Preprocessing pipeline created")
    return preprocessor


def prepare_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split data into features and target.
    
    Args:
        df: DataFrame containing features and target.
        
    Returns:
        Tuple of (features_df, target_series)
    """
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in data")
    
    # Select available feature columns
    available_features = [col for col in FEATURE_COLUMNS if col in df.columns]
    
    if not available_features:
        # If specific feature columns not available, use all except target
        available_features = [col for col in df.columns if col != TARGET_COLUMN]
        logger.warning("Using all columns except target as features")
    
    X = df[available_features]
    y = df[TARGET_COLUMN]
    
    logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
    return X, y


def save_processed_data(df: pd.DataFrame, filename: str) -> None:
    """
    Save processed data to the processed data directory.
    
    Args:
        df: DataFrame to save.
        filename: Name of the output file.
    """
    output_path = PROCESSED_DATA_PATH / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    logger.info(f"Processed data saved to: {output_path}")


def load_processed_data(filename: str) -> pd.DataFrame:
    """
    Load processed data from the processed data directory.
    
    Args:
        filename: Name of the processed data file.
        
    Returns:
        DataFrame with processed data.
    """
    file_path = PROCESSED_DATA_PATH / filename
    
    if not file_path.exists():
        raise FileNotFoundError(f"Processed data file not found: {file_path}")
    
    df = pd.read_csv(file_path)
    logger.info(f"Loaded processed data with shape: {df.shape}")
    return df