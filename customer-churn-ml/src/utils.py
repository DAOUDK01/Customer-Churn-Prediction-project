"""
Utility functions for Customer Churn ML project.

Contains helper functions used across the project.
"""

import pandas as pd
import numpy as np
import logging
import json
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


def setup_logging(log_level: str = 'INFO', log_file: Optional[str] = None) -> None:
    """
    Set up logging configuration for the project.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Path to log file. If None, logs only to console.
    """
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(),  # Console handler
            *([] if log_file is None else [logging.FileHandler(log_file)])  # File handler
        ]
    )
    
    logger.info(f"Logging configured with level: {log_level}")


def save_json(data: Dict[str, Any], file_path: Union[str, Path]) -> None:
    """
    Save dictionary to JSON file.
    
    Args:
        data: Dictionary to save.
        file_path: Path to save the JSON file.
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4, default=str)
    
    logger.info(f"Data saved to JSON: {file_path}")


def load_json(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load dictionary from JSON file.
    
    Args:
        file_path: Path to JSON file.
        
    Returns:
        Loaded dictionary.
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    logger.info(f"Data loaded from JSON: {file_path}")
    return data


def calculate_memory_usage(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate memory usage statistics for a DataFrame.
    
    Args:
        df: DataFrame to analyze.
        
    Returns:
        Dictionary with memory usage information.
    """
    memory_usage = df.memory_usage(deep=True)
    
    stats = {
        'total_memory_mb': memory_usage.sum() / (1024 * 1024),
        'memory_per_column': (memory_usage / (1024 * 1024)).to_dict(),
        'shape': df.shape,
        'dtypes': df.dtypes.to_dict()
    }
    
    logger.info(f"Memory usage calculated: {stats['total_memory_mb']:.2f} MB")
    return stats


def optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize DataFrame memory usage by downcasting numeric types.
    
    Args:
        df: DataFrame to optimize.
        
    Returns:
        Optimized DataFrame.
    """
    df_optimized = df.copy()
    
    # Optimize integer columns
    int_columns = df_optimized.select_dtypes(include=['int64']).columns
    for col in int_columns:
        df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='integer')
    
    # Optimize float columns
    float_columns = df_optimized.select_dtypes(include=['float64']).columns
    for col in float_columns:
        df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='float')
    
    # Convert object columns to category if cardinality is low
    object_columns = df_optimized.select_dtypes(include=['object']).columns
    for col in object_columns:
        unique_count = df_optimized[col].nunique()
        if unique_count < len(df_optimized) * 0.5:  # If unique values < 50% of total
            df_optimized[col] = df_optimized[col].astype('category')
    
    # Calculate memory savings
    original_memory = df.memory_usage(deep=True).sum() / (1024 * 1024)
    optimized_memory = df_optimized.memory_usage(deep=True).sum() / (1024 * 1024)
    memory_reduction = ((original_memory - optimized_memory) / original_memory) * 100
    
    logger.info(f"Memory optimized: {original_memory:.2f} MB -> {optimized_memory:.2f} MB "
                f"({memory_reduction:.1f}% reduction)")
    
    return df_optimized


def get_data_quality_report(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate comprehensive data quality report.
    
    Args:
        df: DataFrame to analyze.
        
    Returns:
        Dictionary with data quality metrics.
    """
    report = {
        'shape': df.shape,
        'missing_values': df.isnull().sum().to_dict(),
        'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
        'duplicated_rows': df.duplicated().sum(),
        'column_dtypes': df.dtypes.to_dict(),
        'unique_values_per_column': df.nunique().to_dict()
    }
    
    # Numeric columns statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        report['numeric_stats'] = df[numeric_cols].describe().to_dict()
    
    # Categorical columns statistics
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        report['categorical_stats'] = {}
        for col in categorical_cols:
            report['categorical_stats'][col] = {
                'unique_count': df[col].nunique(),
                'top_values': df[col].value_counts().head().to_dict()
            }
    
    logger.info("Data quality report generated")
    return report


def plot_data_distribution(df: pd.DataFrame, columns: Optional[List[str]] = None,
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot distribution of numeric columns in a DataFrame.
    
    Args:
        df: DataFrame to plot.
        columns: Specific columns to plot. If None, plots all numeric columns.
        save_path: Path to save the plot (optional).
        
    Returns:
        Matplotlib figure object.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not columns:
        raise ValueError("No numeric columns found for plotting")
    
    n_cols = min(3, len(columns))
    n_rows = (len(columns) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes
    else:
        axes = axes.flatten()
    
    for i, col in enumerate(columns):
        ax = axes[i] if len(columns) > 1 else axes[0]
        
        # Plot histogram with KDE
        df[col].hist(bins=30, alpha=0.7, ax=ax, density=True)
        df[col].plot(kind='kde', ax=ax, color='red')
        
        ax.set_title(f'Distribution of {col}')
        ax.set_xlabel(col)
        ax.set_ylabel('Density')
        ax.grid(True, alpha=0.3)
    
    # Hide empty subplots
    for j in range(len(columns), len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Distribution plot saved to: {save_path}")
    
    return fig


def create_correlation_heatmap(df: pd.DataFrame, save_path: Optional[str] = None) -> plt.Figure:
    """
    Create correlation heatmap for numeric columns.
    
    Args:
        df: DataFrame to analyze.
        save_path: Path to save the plot (optional).
        
    Returns:
        Matplotlib figure object.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.empty:
        raise ValueError("No numeric columns found for correlation analysis")
    
    correlation_matrix = numeric_df.corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.1, cbar_kws={"shrink": 0.8}, ax=ax)
    
    ax.set_title('Feature Correlation Heatmap')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Correlation heatmap saved to: {save_path}")
    
    return fig


def detect_outliers_iqr(df: pd.DataFrame, columns: Optional[List[str]] = None) -> Dict[str, List]:
    """
    Detect outliers using the IQR method.
    
    Args:
        df: DataFrame to analyze.
        columns: Columns to check for outliers. If None, checks all numeric columns.
        
    Returns:
        Dictionary with outlier indices for each column.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    outliers = {}
    
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outlier_indices = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index.tolist()
        outliers[col] = outlier_indices
        
        logger.info(f"Found {len(outlier_indices)} outliers in column '{col}'")
    
    return outliers


def encode_categorical_features(df: pd.DataFrame, encoding_method: str = 'onehot',
                              categorical_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Encode categorical features using specified method.
    
    Args:
        df: DataFrame with categorical features.
        encoding_method: 'onehot', 'label', or 'target'.
        categorical_columns: Columns to encode. If None, auto-detects categorical columns.
        
    Returns:
        DataFrame with encoded features.
    """
    df_encoded = df.copy()
    
    if categorical_columns is None:
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if encoding_method == 'onehot':
        df_encoded = pd.get_dummies(df_encoded, columns=categorical_columns, 
                                  drop_first=True, prefix=categorical_columns)
    
    elif encoding_method == 'label':
        from sklearn.preprocessing import LabelEncoder
        
        for col in categorical_columns:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    
    else:
        raise ValueError(f"Unsupported encoding method: {encoding_method}")
    
    logger.info(f"Categorical encoding completed using {encoding_method} method")
    return df_encoded


def validate_model_input(X: pd.DataFrame, expected_columns: Optional[List[str]] = None) -> bool:
    """
    Validate input data for model prediction.
    
    Args:
        X: Input features DataFrame.
        expected_columns: Expected column names. If None, skips column validation.
        
    Returns:
        True if validation passes.
        
    Raises:
        ValueError: If validation fails.
    """
    # Check for empty DataFrame
    if X.empty:
        raise ValueError("Input DataFrame is empty")
    
    # Check for missing values
    if X.isnull().any().any():
        missing_cols = X.columns[X.isnull().any()].tolist()
        raise ValueError(f"Missing values found in columns: {missing_cols}")
    
    # Check column names if provided
    if expected_columns is not None:
        missing_cols = set(expected_columns) - set(X.columns)
        if missing_cols:
            raise ValueError(f"Missing expected columns: {missing_cols}")
        
        extra_cols = set(X.columns) - set(expected_columns)
        if extra_cols:
            logger.warning(f"Extra columns found (will be ignored): {extra_cols}")
    
    logger.info("Input validation passed")
    return True


def format_model_metrics(metrics: Dict[str, float], precision: int = 4) -> str:
    """
    Format model metrics for nice display.
    
    Args:
        metrics: Dictionary of metric names and values.
        precision: Number of decimal places to display.
        
    Returns:
        Formatted string with metrics.
    """
    formatted_lines = []
    
    for metric_name, value in metrics.items():
        # Convert metric name to title case and replace underscores
        display_name = metric_name.replace('_', ' ').title()
        formatted_lines.append(f"{display_name}: {value:.{precision}f}")
    
    return "\n".join(formatted_lines)