"""
Model evaluation utilities for Customer Churn ML project.

Contains functions for evaluating classification model performance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, Any, Optional, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.model_selection import cross_val_score

from .config import REPORTS_PATH

logger = logging.getLogger(__name__)


def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                                   y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Calculate comprehensive classification metrics.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        y_pred_proba: Predicted probabilities (optional, for AUC calculation).
        
    Returns:
        Dictionary with various classification metrics.
    """
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average='weighted')
    metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
    metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted')
    
    # Class-specific metrics
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro')
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro')
    metrics['f1_score_macro'] = f1_score(y_true, y_pred, average='macro')
    
    # AUC metrics (if probabilities provided)
    if y_pred_proba is not None:
        try:
            if len(np.unique(y_true)) == 2:  # Binary classification
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
                metrics['pr_auc'] = average_precision_score(y_true, y_pred_proba)
            else:  # Multi-class
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
        except ValueError as e:
            logger.warning(f"Could not calculate AUC metrics: {str(e)}")
    
    logger.info("Classification metrics calculated successfully")
    return metrics


def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray, 
                              target_names: Optional[list] = None) -> str:
    """
    Generate and print detailed classification report.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        target_names: Names of target classes (optional).
        
    Returns:
        Classification report as string.
    """
    report = classification_report(y_true, y_pred, target_names=target_names)
    print("Classification Report:")
    print("=" * 50)
    print(report)
    return report


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                         target_names: Optional[list] = None, 
                         save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot confusion matrix with nice formatting.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        target_names: Names of target classes (optional).
        save_path: Path to save the plot (optional).
        
    Returns:
        Matplotlib figure object.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names or ['Class 0', 'Class 1'],
                yticklabels=target_names or ['Class 0', 'Class 1'],
                ax=ax)
    
    ax.set_title('Confusion Matrix')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    
    plt.tight_layout()
    
    if save_path:
        save_path = REPORTS_PATH / save_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to: {save_path}")
    
    return fig


def plot_roc_curve(y_true: np.ndarray, y_pred_proba: np.ndarray, 
                   save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot ROC curve for binary classification.
    
    Args:
        y_true: True binary labels.
        y_pred_proba: Predicted probabilities for positive class.
        save_path: Path to save the plot (optional).
        
    Returns:
        Matplotlib figure object.
    """
    if len(np.unique(y_true)) != 2:
        raise ValueError("ROC curve is only applicable for binary classification")
    
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    auc_score = roc_auc_score(y_true, y_pred_proba)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC curve (AUC = {auc_score:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.8)
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_path = REPORTS_PATH / save_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"ROC curve saved to: {save_path}")
    
    return fig


def plot_precision_recall_curve(y_true: np.ndarray, y_pred_proba: np.ndarray, 
                               save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot Precision-Recall curve for binary classification.
    
    Args:
        y_true: True binary labels.
        y_pred_proba: Predicted probabilities for positive class.
        save_path: Path to save the plot (optional).
        
    Returns:
        Matplotlib figure object.
    """
    if len(np.unique(y_true)) != 2:
        raise ValueError("PR curve is only applicable for binary classification")
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    avg_precision = average_precision_score(y_true, y_pred_proba)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(recall, precision, color='darkorange', lw=2,
            label=f'PR curve (Average Precision = {avg_precision:.3f})')
    ax.axhline(y=y_true.mean(), color='navy', linestyle='--', alpha=0.8,
               label=f'Baseline (Positive Rate = {y_true.mean():.3f})')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_path = REPORTS_PATH / save_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Precision-Recall curve saved to: {save_path}")
    
    return fig


def cross_validate_model(model, X: np.ndarray, y: np.ndarray, 
                        cv: int = 5, scoring: str = 'accuracy') -> Dict[str, Any]:
    """
    Perform cross-validation on a model.
    
    Args:
        model: Scikit-learn model or pipeline.
        X: Feature matrix.
        y: Target vector.
        cv: Number of cross-validation folds.
        scoring: Scoring metric for cross-validation.
        
    Returns:
        Dictionary with cross-validation results.
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    
    cv_results = {
        'scores': scores,
        'mean_score': scores.mean(),
        'std_score': scores.std(),
        'min_score': scores.min(),
        'max_score': scores.max()
    }
    
    logger.info(f"Cross-validation completed. Mean {scoring}: {cv_results['mean_score']:.4f} "
                f"(+/- {cv_results['std_score'] * 2:.4f})")
    
    return cv_results


def feature_importance_plot(feature_names: list, importance_values: np.ndarray,
                          save_path: Optional[str] = None, top_n: int = 20) -> plt.Figure:
    """
    Plot feature importance with horizontal bar chart.
    
    Args:
        feature_names: Names of features.
        importance_values: Importance values for each feature.
        save_path: Path to save the plot (optional).
        top_n: Number of top features to display.
        
    Returns:
        Matplotlib figure object.
    """
    # Create DataFrame and sort by importance
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_values
    }).sort_values('importance', ascending=True)
    
    # Select top N features
    top_features = importance_df.tail(top_n)
    
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.4)))
    
    bars = ax.barh(top_features['feature'], top_features['importance'], 
                   color='skyblue', edgecolor='navy', alpha=0.7)
    
    ax.set_xlabel('Importance')
    ax.set_title(f'Top {top_n} Feature Importance')
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', ha='left', va='center', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        save_path = REPORTS_PATH / save_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Feature importance plot saved to: {save_path}")
    
    return fig


def generate_model_report(y_true: np.ndarray, y_pred: np.ndarray, 
                         y_pred_proba: Optional[np.ndarray] = None,
                         target_names: Optional[list] = None,
                         model_name: str = "Model") -> Dict[str, Any]:
    """
    Generate comprehensive model evaluation report.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        y_pred_proba: Predicted probabilities (optional).
        target_names: Names of target classes (optional).
        model_name: Name of the model for reporting.
        
    Returns:
        Dictionary containing all evaluation results.
    """
    logger.info(f"Generating evaluation report for {model_name}")
    
    report = {
        'model_name': model_name,
        'metrics': calculate_classification_metrics(y_true, y_pred, y_pred_proba),
        'classification_report': classification_report(y_true, y_pred, 
                                                     target_names=target_names, 
                                                     output_dict=True)
    }
    
    # Print summary
    print(f"\n{model_name} Evaluation Report")
    print("=" * 50)
    print(f"Accuracy: {report['metrics']['accuracy']:.4f}")
    print(f"Precision: {report['metrics']['precision']:.4f}")
    print(f"Recall: {report['metrics']['recall']:.4f}")
    print(f"F1-Score: {report['metrics']['f1_score']:.4f}")
    
    if 'roc_auc' in report['metrics']:
        print(f"ROC AUC: {report['metrics']['roc_auc']:.4f}")
    
    return report