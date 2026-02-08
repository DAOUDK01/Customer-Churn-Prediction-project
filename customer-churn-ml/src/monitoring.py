"""
Production monitoring utilities for Customer Churn ML project.

Handles data drift detection, prediction monitoring, and system health checks
for production ML systems.
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import logging
from scipy import stats
from scipy.stats import ks_2samp
import warnings

from .config import REPORTS_PATH

logger = logging.getLogger(__name__)


class DataDriftMonitor:
    """Monitor for data drift in production inference data."""
    
    def __init__(self, reference_data: pd.DataFrame, 
                 numeric_threshold: float = 0.05,
                 categorical_threshold: float = 0.05):
        """Initialize drift monitor with reference data.
        
        Args:
            reference_data: Training/reference dataset
            numeric_threshold: P-value threshold for numeric drift detection
            categorical_threshold: P-value threshold for categorical drift
        """
        self.reference_data = reference_data
        self.numeric_threshold = numeric_threshold
        self.categorical_threshold = categorical_threshold
        
        # Compute reference statistics
        self._compute_reference_stats()
        
    def _compute_reference_stats(self) -> None:
        """Compute reference statistics for drift comparison."""
        self.reference_stats = {}
        
        for column in self.reference_data.columns:
            if pd.api.types.is_numeric_dtype(self.reference_data[column]):
                self.reference_stats[column] = {
                    'type': 'numeric',
                    'mean': self.reference_data[column].mean(),
                    'std': self.reference_data[column].std(),
                    'quantiles': self.reference_data[column].quantile([0.25, 0.5, 0.75]).to_dict()
                }
            else:
                value_counts = self.reference_data[column].value_counts(normalize=True)
                self.reference_stats[column] = {
                    'type': 'categorical',
                    'distribution': value_counts.to_dict(),
                    'unique_count': len(value_counts)
                }
    
    def detect_drift(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect data drift between reference and current data.
        
        Args:
            current_data: Current/inference dataset
            
        Returns:
            Dictionary with drift detection results
        """
        drift_results = {
            'timestamp': datetime.now().isoformat(),
            'total_samples': len(current_data),
            'columns_analyzed': 0,
            'drift_detected': False,
            'drifted_columns': [],
            'column_results': {}
        }
        
        for column in self.reference_data.columns:
            if column not in current_data.columns:
                continue
                
            drift_results['columns_analyzed'] += 1
            
            if self.reference_stats[column]['type'] == 'numeric':
                result = self._test_numeric_drift(
                    self.reference_data[column], 
                    current_data[column],
                    column
                )
            else:
                result = self._test_categorical_drift(
                    self.reference_data[column],
                    current_data[column], 
                    column
                )
            
            drift_results['column_results'][column] = result
            
            if result['drift_detected']:
                drift_results['drift_detected'] = True
                drift_results['drifted_columns'].append(column)
        
        # Overall drift score (percentage of drifted columns)
        drift_results['drift_score'] = len(drift_results['drifted_columns']) / drift_results['columns_analyzed']
        
        return drift_results
    
    def _test_numeric_drift(self, reference: pd.Series, 
                          current: pd.Series, 
                          column: str) -> Dict[str, Any]:
        """Test for drift in numeric columns using Kolmogorov-Smirnov test.
        
        Args:
            reference: Reference data column
            current: Current data column
            column: Column name
            
        Returns:
            Drift test results
        """
        # Remove missing values
        ref_clean = reference.dropna()
        curr_clean = current.dropna()
        
        if len(curr_clean) == 0:
            return {
                'drift_detected': True,
                'test': 'KS test',
                'p_value': 0.0,
                'statistic': 1.0,
                'reason': 'All values missing in current data'
            }
        
        # Kolmogorov-Smirnov test
        ks_stat, p_value = ks_2samp(ref_clean, curr_clean)
        
        # Additional statistics
        ref_mean = ref_clean.mean()
        curr_mean = curr_clean.mean()
        mean_shift = abs(curr_mean - ref_mean) / ref_clean.std() if ref_clean.std() > 0 else 0
        
        return {
            'drift_detected': p_value < self.numeric_threshold,
            'test': 'KS test',
            'p_value': float(p_value),
            'statistic': float(ks_stat),
            'mean_shift': float(mean_shift),
            'reference_mean': float(ref_mean),
            'current_mean': float(curr_mean),
            'missing_rate_change': float(current.isnull().mean() - reference.isnull().mean())
        }
    
    def _test_categorical_drift(self, reference: pd.Series,
                              current: pd.Series,
                              column: str) -> Dict[str, Any]:
        """Test for drift in categorical columns using Chi-square test.
        
        Args:
            reference: Reference data column  
            current: Current data column
            column: Column name
            
        Returns:
            Drift test results
        """
        # Get value distributions
        ref_dist = reference.value_counts(normalize=True, dropna=False)
        curr_dist = current.value_counts(normalize=True, dropna=False)
        
        # Align distributions (fill missing categories with 0)
        all_categories = set(ref_dist.index) | set(curr_dist.index)
        ref_aligned = ref_dist.reindex(all_categories, fill_value=0)
        curr_aligned = curr_dist.reindex(all_categories, fill_value=0)
        
        # Chi-square test for goodness of fit
        observed = curr_aligned * len(current)
        expected = ref_aligned * len(current)
        
        # Avoid division by zero
        expected = expected.replace(0, 1e-10)
        
        chi2_stat = np.sum((observed - expected) ** 2 / expected)
        df = len(all_categories) - 1
        p_value = 1 - stats.chi2.cdf(chi2_stat, df) if df > 0 else 1.0
        
        # Calculate distribution distance (Total Variation Distance)
        tv_distance = 0.5 * np.sum(np.abs(ref_aligned - curr_aligned))
        
        # New categories
        new_categories = set(curr_dist.index) - set(ref_dist.index)
        
        return {
            'drift_detected': p_value < self.categorical_threshold,
            'test': 'Chi-square',
            'p_value': float(p_value),
            'statistic': float(chi2_stat),
            'tv_distance': float(tv_distance),
            'new_categories': list(new_categories),
            'missing_rate_change': float(current.isnull().mean() - reference.isnull().mean())
        }


class PredictionMonitor:
    """Monitor prediction distributions and model behavior."""
    
    def __init__(self, log_file: Optional[Path] = None):
        """Initialize prediction monitor.
        
        Args:
            log_file: Path to prediction log file
        """
        self.log_file = log_file or REPORTS_PATH / "prediction_log.json"
        self.log_file.parent.mkdir(exist_ok=True, parents=True)
        
    def log_predictions(self, 
                       predictions: np.ndarray,
                       probabilities: Optional[np.ndarray] = None,
                       input_data: Optional[pd.DataFrame] = None,
                       model_version: Optional[str] = None) -> None:
        """Log prediction results for monitoring.
        
        Args:
            predictions: Model predictions
            probabilities: Model prediction probabilities
            input_data: Input feature data
            model_version: Model version used
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'model_version': model_version,
            'batch_size': len(predictions),
            'prediction_stats': {
                'positive_rate': float(np.mean(predictions)),
                'negative_rate': float(1 - np.mean(predictions)),
                'unique_predictions': len(np.unique(predictions))
            }
        }
        
        if probabilities is not None:
            log_entry['probability_stats'] = {
                'mean_probability': float(np.mean(probabilities)),
                'std_probability': float(np.std(probabilities)),
                'min_probability': float(np.min(probabilities)),
                'max_probability': float(np.max(probabilities)),
                'high_confidence_rate': float(np.mean(probabilities > 0.8)),
                'low_confidence_rate': float(np.mean(probabilities < 0.2))
            }
        
        if input_data is not None:
            log_entry['input_stats'] = {
                'feature_count': len(input_data.columns),
                'missing_rate': float(input_data.isnull().mean().mean()),
                'numeric_feature_means': input_data.select_dtypes(include=[np.number]).mean().to_dict()
            }
        
        # Append to log file
        self._append_log(log_entry)
        
    def _append_log(self, log_entry: Dict[str, Any]) -> None:
        """Append log entry to file."""
        try:
            # Read existing logs
            if self.log_file.exists():
                with open(self.log_file, 'r') as f:
                    logs = json.load(f)
            else:
                logs = []
            
            # Append new entry
            logs.append(log_entry)
            
            # Keep only last 1000 entries to manage file size
            logs = logs[-1000:]
            
            # Write back
            with open(self.log_file, 'w') as f:
                json.dump(logs, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to write prediction log: {e}")
    
    def get_recent_stats(self, hours: int = 24) -> Dict[str, Any]:
        """Get prediction statistics from recent time window.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            Aggregated prediction statistics
        """
        if not self.log_file.exists():
            return {}
        
        try:
            with open(self.log_file, 'r') as f:
                logs = json.load(f)
            
            # Filter to recent logs
            cutoff_time = datetime.now().timestamp() - (hours * 3600)
            recent_logs = [
                log for log in logs 
                if datetime.fromisoformat(log['timestamp']).timestamp() > cutoff_time
            ]
            
            if not recent_logs:
                return {}
            
            # Aggregate statistics
            total_predictions = sum(log['batch_size'] for log in recent_logs)
            avg_positive_rate = np.mean([log['prediction_stats']['positive_rate'] for log in recent_logs])
            
            stats = {
                'time_window_hours': hours,
                'total_predictions': total_predictions,
                'batch_count': len(recent_logs),
                'average_positive_rate': avg_positive_rate,
                'prediction_volume_trend': self._calculate_trend(recent_logs)
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to read prediction stats: {e}")
            return {}
    
    def _calculate_trend(self, logs: List[Dict[str, Any]]) -> str:
        """Calculate trend in prediction volume."""
        if len(logs) < 2:
            return "insufficient_data"
        
        # Simple trend calculation based on batch sizes
        batch_sizes = [log['batch_size'] for log in logs]
        first_half = np.mean(batch_sizes[:len(batch_sizes)//2])
        second_half = np.mean(batch_sizes[len(batch_sizes)//2:])
        
        if second_half > first_half * 1.1:
            return "increasing"
        elif second_half < first_half * 0.9:
            return "decreasing"
        else:
            return "stable"


class SystemHealthMonitor:
    """Monitor overall system health and alert on issues."""
    
    def __init__(self):
        """Initialize system health monitor."""
        self.checks = {
            'model_available': self._check_model_available,
            'data_quality': self._check_data_quality,
            'prediction_distribution': self._check_prediction_distribution,
            'performance_degradation': self._check_performance_degradation
        }
    
    def run_health_check(self, 
                        model_path: Optional[Path] = None,
                        recent_data: Optional[pd.DataFrame] = None,
                        recent_predictions: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Run comprehensive health check.
        
        Args:
            model_path: Path to model file
            recent_data: Recent input data
            recent_predictions: Recent predictions
            
        Returns:
            Health check results
        """
        health_status = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'healthy',
            'checks': {},
            'alerts': []
        }
        
        # Run all health checks
        for check_name, check_func in self.checks.items():
            try:
                result = check_func(model_path, recent_data, recent_predictions)
                health_status['checks'][check_name] = result
                
                if not result['passed']:
                    health_status['alerts'].append({
                        'check': check_name,
                        'message': result['message'],
                        'severity': result.get('severity', 'warning')
                    })
                    
            except Exception as e:
                health_status['checks'][check_name] = {
                    'passed': False,
                    'message': f"Check failed with error: {str(e)}",
                    'severity': 'error'
                }
        
        # Determine overall status
        if any(alert['severity'] == 'error' for alert in health_status['alerts']):
            health_status['overall_status'] = 'unhealthy'
        elif health_status['alerts']:
            health_status['overall_status'] = 'degraded'
        
        return health_status
    
    def _check_model_available(self, model_path, recent_data, recent_predictions) -> Dict[str, Any]:
        """Check if model is available and loadable."""
        try:
            if model_path and model_path.exists():
                # Try to load the model
                import joblib
                joblib.load(model_path)
                return {
                    'passed': True,
                    'message': 'Model is available and loadable'
                }
            else:
                return {
                    'passed': False,
                    'message': 'Model file not found or not accessible',
                    'severity': 'error'
                }
        except Exception as e:
            return {
                'passed': False,
                'message': f'Model loading failed: {str(e)}',
                'severity': 'error'
            }
    
    def _check_data_quality(self, model_path, recent_data, recent_predictions) -> Dict[str, Any]:
        """Check recent data quality."""
        if recent_data is None:
            return {
                'passed': True,
                'message': 'No recent data to check'
            }
        
        missing_rate = recent_data.isnull().mean().mean()
        duplicate_rate = recent_data.duplicated().mean()
        
        if missing_rate > 0.5:
            return {
                'passed': False,
                'message': f'High missing data rate: {missing_rate:.1%}',
                'severity': 'warning'
            }
        elif duplicate_rate > 0.1:
            return {
                'passed': False,
                'message': f'High duplicate rate: {duplicate_rate:.1%}',
                'severity': 'warning'
            }
        else:
            return {
                'passed': True,
                'message': 'Data quality within acceptable bounds'
            }
    
    def _check_prediction_distribution(self, model_path, recent_data, recent_predictions) -> Dict[str, Any]:
        """Check if prediction distribution is reasonable."""
        if recent_predictions is None:
            return {
                'passed': True,
                'message': 'No recent predictions to check'
            }
        
        positive_rate = np.mean(recent_predictions)
        
        # Expected churn rate is around 10-30% for most businesses
        if positive_rate < 0.05 or positive_rate > 0.5:
            return {
                'passed': False,
                'message': f'Unusual prediction distribution: {positive_rate:.1%} positive rate',
                'severity': 'warning'
            }
        else:
            return {
                'passed': True,
                'message': f'Prediction distribution normal: {positive_rate:.1%} positive rate'
            }
    
    def _check_performance_degradation(self, model_path, recent_data, recent_predictions) -> Dict[str, Any]:
        """Check for signs of performance degradation."""
        # This would typically involve comparing recent performance to baseline
        # For now, return a simple check
        return {
            'passed': True,
            'message': 'Performance monitoring not implemented yet'
        }