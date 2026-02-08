"""
Advanced Model Monitoring and Drift Detection System
Enterprise-grade monitoring with real-time alerts and automated retraining triggers
"""

import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
import logging
from scipy import stats
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib
import warnings
warnings.filterwarnings('ignore')

try:\n    from src.config import PROJECT_ROOT, MONITORING_CONFIG\nexcept ImportError:\n    from config import PROJECT_ROOT, MONITORING_CONFIG

logger = logging.getLogger(__name__)

class AdvancedModelMonitor:
    """
    Advanced monitoring system for production ML models
    Includes data drift, model performance, and business impact monitoring
    """
    
    def __init__(self, model_path: str = None, reference_data_path: str = None):
        self.model_path = model_path
        self.reference_data_path = reference_data_path
        self.monitoring_data = []
        self.alerts = []
        self.baseline_stats = {}
        self.performance_history = []
        
        # Monitoring thresholds
        self.drift_threshold = 0.1  # PSI threshold for drift detection
        self.performance_threshold = 0.05  # Performance degradation threshold
        self.alert_threshold = 0.15  # Critical alert threshold
        
        # Initialize monitoring
        self._setup_monitoring()
    
    def _setup_monitoring(self):
        """Initialize monitoring system and baseline statistics"""
        try:
            # Load reference data if available
            if self.reference_data_path and Path(self.reference_data_path).exists():
                reference_data = pd.read_csv(self.reference_data_path)
                self._calculate_baseline_statistics(reference_data)
                logger.info("✅ Baseline statistics calculated from reference data")
            else:
                logger.warning("⚠️ No reference data found - using default baselines")
                self._set_default_baselines()
                
        except Exception as e:
            logger.error(f"❌ Failed to setup monitoring: {e}")
            self._set_default_baselines()
    
    def _calculate_baseline_statistics(self, reference_data: pd.DataFrame):
        """Calculate baseline statistics from reference data"""
        numeric_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']
        
        for col in numeric_columns:
            if col in reference_data.columns:
                self.baseline_stats[col] = {
                    'mean': reference_data[col].mean(),
                    'std': reference_data[col].std(),
                    'min': reference_data[col].min(),
                    'max': reference_data[col].max(),
                    'percentiles': reference_data[col].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict()
                }
    
    def _set_default_baselines(self):
        """Set default baseline statistics"""
        self.baseline_stats = {
            'tenure': {
                'mean': 32.0, 'std': 24.0, 'min': 0, 'max': 72,
                'percentiles': {0.1: 1, 0.25: 9, 0.5: 29, 0.75: 55, 0.9: 70}
            },
            'MonthlyCharges': {
                'mean': 64.8, 'std': 30.1, 'min': 18.3, 'max': 118.7,
                'percentiles': {0.1: 20, 0.25: 35, 0.5: 70, 0.75: 89, 0.9: 105}
            },
            'TotalCharges': {
                'mean': 2283.3, 'std': 2266.8, 'min': 18.8, 'max': 8684.8,
                'percentiles': {0.1: 20, 0.25: 400, 0.5: 1400, 0.75: 3800, 0.9: 6000}
            }
        }
    
    def monitor_prediction_batch(self, features: pd.DataFrame, predictions: np.ndarray, 
                                actual_outcomes: np.ndarray = None) -> Dict[str, Any]:
        """
        Monitor a batch of predictions for drift and performance issues
        """
        monitoring_result = {
            'timestamp': datetime.now().isoformat(),
            'batch_size': len(features),
            'drift_detected': False,
            'performance_degraded': False,
            'alerts': [],
            'metrics': {}
        }
        
        try:
            # 1. Data Drift Detection
            drift_results = self._detect_data_drift(features)
            monitoring_result['drift_detected'] = drift_results['drift_detected']
            monitoring_result['metrics']['drift'] = drift_results
            
            # 2. Prediction Distribution Analysis
            prediction_analysis = self._analyze_prediction_distribution(predictions)
            monitoring_result['metrics']['predictions'] = prediction_analysis
            
            # 3. Performance Monitoring (if actual outcomes available)
            if actual_outcomes is not None:
                performance_results = self._monitor_performance(predictions, actual_outcomes)
                monitoring_result['performance_degraded'] = performance_results['degraded']
                monitoring_result['metrics']['performance'] = performance_results
            
            # 4. Generate alerts
            alerts = self._generate_alerts(monitoring_result['metrics'])
            monitoring_result['alerts'] = alerts
            
            # 5. Store monitoring data
            self.monitoring_data.append(monitoring_result)
            
            # 6. Check for critical issues
            if monitoring_result['drift_detected'] or monitoring_result['performance_degraded']:
                self._trigger_alerts(monitoring_result)
            
            return monitoring_result
            
        except Exception as e:
            logger.error(f"❌ Monitoring failed: {e}")
            return self._create_error_result(str(e))
    
    def _detect_data_drift(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect data drift using Population Stability Index (PSI)"""
        drift_results = {
            'drift_detected': False,
            'drift_scores': {},
            'drifted_features': []
        }
        
        try:
            for feature in ['tenure', 'MonthlyCharges', 'TotalCharges']:
                if feature in current_data.columns and feature in self.baseline_stats:
                    psi_score = self._calculate_psi(
                        current_data[feature], 
                        self.baseline_stats[feature]
                    )
                    
                    drift_results['drift_scores'][feature] = psi_score
                    
                    if psi_score > self.drift_threshold:
                        drift_results['drifted_features'].append(feature)
                        drift_results['drift_detected'] = True
            
            return drift_results
            
        except Exception as e:
            logger.error(f"❌ Drift detection failed: {e}")
            return {'drift_detected': False, 'error': str(e)}
    
    def _calculate_psi(self, current_data: pd.Series, baseline_stats: Dict) -> float:
        """Calculate Population Stability Index (PSI)"""
        try:
            # Create bins based on baseline percentiles
            percentiles = baseline_stats['percentiles']
            bins = [baseline_stats['min']] + list(percentiles.values()) + [baseline_stats['max']]
            bins = sorted(list(set(bins)))  # Remove duplicates and sort
            
            # Calculate expected (baseline) and actual (current) distributions
            expected_counts = [1/len(bins) for _ in range(len(bins)-1)]  # Uniform distribution assumption
            
            # Calculate current distribution
            current_counts, _ = np.histogram(current_data.dropna(), bins=bins)
            current_proportions = current_counts / current_counts.sum()
            
            # Calculate PSI
            psi = 0
            for i in range(len(expected_counts)):
                expected_prop = expected_counts[i]
                actual_prop = current_proportions[i] if i < len(current_proportions) else 0.001
                
                # Avoid division by zero
                if expected_prop == 0:
                    expected_prop = 0.001
                if actual_prop == 0:
                    actual_prop = 0.001
                
                psi += (actual_prop - expected_prop) * np.log(actual_prop / expected_prop)
            
            return abs(psi)
            
        except Exception as e:
            logger.error(f"❌ PSI calculation failed: {e}")
            return 0.0
    
    def _analyze_prediction_distribution(self, predictions: np.ndarray) -> Dict[str, Any]:
        """Analyze the distribution of predictions"""
        try:
            # Convert probabilities to risk categories
            high_risk_count = sum(1 for p in predictions if p >= 0.7)
            medium_risk_count = sum(1 for p in predictions if 0.3 <= p < 0.7)
            low_risk_count = sum(1 for p in predictions if p < 0.3)
            
            total_predictions = len(predictions)
            
            return {
                'total_predictions': total_predictions,
                'high_risk_rate': high_risk_count / total_predictions,
                'medium_risk_rate': medium_risk_count / total_predictions,
                'low_risk_rate': low_risk_count / total_predictions,
                'avg_prediction': np.mean(predictions),
                'prediction_std': np.std(predictions),
                'prediction_percentiles': {
                    'p10': np.percentile(predictions, 10),
                    'p25': np.percentile(predictions, 25),
                    'p50': np.percentile(predictions, 50),
                    'p75': np.percentile(predictions, 75),
                    'p90': np.percentile(predictions, 90)
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Prediction analysis failed: {e}")
            return {'error': str(e)}
    
    def _monitor_performance(self, predictions: np.ndarray, actual_outcomes: np.ndarray) -> Dict[str, Any]:
        """Monitor model performance against actual outcomes"""
        try:
            # Convert probabilities to binary predictions
            binary_predictions = (predictions >= 0.5).astype(int)
            
            # Calculate performance metrics
            accuracy = accuracy_score(actual_outcomes, binary_predictions)
            precision = precision_score(actual_outcomes, binary_predictions, zero_division=0)
            recall = recall_score(actual_outcomes, binary_predictions, zero_division=0)
            
            performance_result = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'degraded': False
            }
            
            # Check against historical performance
            if self.performance_history:
                recent_accuracy = np.mean([p['accuracy'] for p in self.performance_history[-10:]])
                if accuracy < recent_accuracy - self.performance_threshold:
                    performance_result['degraded'] = True
                    performance_result['degradation_amount'] = recent_accuracy - accuracy
            
            # Store performance history
            self.performance_history.append({
                'timestamp': datetime.now().isoformat(),
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall
            })
            
            return performance_result
            
        except Exception as e:
            logger.error(f"❌ Performance monitoring failed: {e}")
            return {'error': str(e), 'degraded': False}
    
    def _generate_alerts(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate alerts based on monitoring metrics"""
        alerts = []
        
        try:
            # Data drift alerts
            if 'drift' in metrics and metrics['drift'].get('drift_detected'):
                drift_features = metrics['drift'].get('drifted_features', [])
                alerts.append({
                    'type': 'DATA_DRIFT',
                    'severity': 'HIGH' if len(drift_features) > 2 else 'MEDIUM',
                    'message': f"Data drift detected in features: {', '.join(drift_features)}",
                    'timestamp': datetime.now().isoformat(),
                    'action_required': 'Consider model retraining'
                })
            
            # Performance degradation alerts
            if 'performance' in metrics and metrics['performance'].get('degraded'):
                degradation = metrics['performance'].get('degradation_amount', 0)
                alerts.append({
                    'type': 'PERFORMANCE_DEGRADATION',
                    'severity': 'HIGH' if degradation > 0.1 else 'MEDIUM',
                    'message': f"Model performance degraded by {degradation:.3f}",
                    'timestamp': datetime.now().isoformat(),
                    'action_required': 'Investigate and potentially retrain model'
                })
            
            # Prediction distribution alerts
            if 'predictions' in metrics:
                high_risk_rate = metrics['predictions'].get('high_risk_rate', 0)
                if high_risk_rate > 0.2:  # More than 20% high risk
                    alerts.append({
                        'type': 'HIGH_RISK_SPIKE',
                        'severity': 'HIGH',
                        'message': f"Unusually high churn risk detected: {high_risk_rate:.1%} high risk customers",
                        'timestamp': datetime.now().isoformat(),
                        'action_required': 'Investigate business conditions and customer behavior'
                    })
            
            return alerts
            
        except Exception as e:
            logger.error(f"❌ Alert generation failed: {e}")
            return [{'type': 'MONITORING_ERROR', 'message': str(e)}]
    
    def _trigger_alerts(self, monitoring_result: Dict[str, Any]):
        """Trigger alerts for critical issues"""
        try:
            for alert in monitoring_result.get('alerts', []):
                if alert.get('severity') == 'HIGH':
                    # In production, integrate with alerting systems (PagerDuty, Slack, etc.)
                    logger.critical(f"🚨 CRITICAL ALERT: {alert['message']}")
                    
                    # Store alert for dashboard
                    self.alerts.append(alert)
                    
                    # Trigger automated responses
                    if alert['type'] == 'DATA_DRIFT':
                        self._schedule_model_retraining()
                    elif alert['type'] == 'PERFORMANCE_DEGRADATION':
                        self._flag_for_immediate_review()
                        
        except Exception as e:
            logger.error(f"❌ Alert triggering failed: {e}")
    
    def _schedule_model_retraining(self):
        """Schedule automated model retraining"""
        logger.info("📅 Scheduling model retraining due to data drift")
        # In production, trigger retraining pipeline
    
    def _flag_for_immediate_review(self):
        """Flag model for immediate manual review"""
        logger.info("🔍 Flagging model for immediate manual review")
        # In production, notify ML engineering team
    
    def get_monitoring_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard"""
        try:
            recent_data = self.monitoring_data[-100:] if self.monitoring_data else []
            
            dashboard_data = {
                'summary': {
                    'total_monitored_batches': len(self.monitoring_data),
                    'active_alerts': len([a for a in self.alerts if self._is_alert_active(a)]),
                    'drift_detected_batches': len([d for d in recent_data if d.get('drift_detected')]),
                    'performance_issues': len([d for d in recent_data if d.get('performance_degraded')])
                },
                'recent_metrics': recent_data,
                'active_alerts': [a for a in self.alerts if self._is_alert_active(a)],
                'performance_trend': self.performance_history[-50:] if self.performance_history else [],
                'system_health': {
                    'status': self._get_system_health_status(),
                    'last_check': datetime.now().isoformat(),
                    'recommendation': self._get_health_recommendation()
                }
            }
            
            return dashboard_data
            
        except Exception as e:
            logger.error(f"❌ Dashboard data generation failed: {e}")
            return {'error': str(e)}
    
    def _is_alert_active(self, alert: Dict[str, Any]) -> bool:
        """Check if alert is still active (within last 24 hours)"""
        try:
            alert_time = datetime.fromisoformat(alert['timestamp'])
            return (datetime.now() - alert_time) < timedelta(hours=24)
        except:
            return False
    
    def _get_system_health_status(self) -> str:
        """Get overall system health status"""
        active_alerts = [a for a in self.alerts if self._is_alert_active(a)]
        high_severity_alerts = [a for a in active_alerts if a.get('severity') == 'HIGH']
        
        if high_severity_alerts:
            return 'CRITICAL'
        elif active_alerts:
            return 'WARNING'
        else:
            return 'HEALTHY'
    
    def _get_health_recommendation(self) -> str:
        """Get health-based recommendations"""
        status = self._get_system_health_status()
        
        if status == 'CRITICAL':
            return 'Immediate action required - investigate alerts and consider model rollback'
        elif status == 'WARNING':
            return 'Monitor closely and prepare for potential intervention'
        else:
            return 'System operating normally'
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create error result for monitoring failures"""
        return {
            'timestamp': datetime.now().isoformat(),
            'error': True,
            'message': error_message,
            'drift_detected': False,
            'performance_degraded': False,
            'alerts': [{
                'type': 'MONITORING_ERROR',
                'severity': 'HIGH',
                'message': f"Monitoring system error: {error_message}",
                'timestamp': datetime.now().isoformat()
            }]
        }

# Factory function for easy instantiation
def create_production_monitor(model_path: str = None, reference_data_path: str = None) -> AdvancedModelMonitor:
    """Create a production-ready model monitor"""
    return AdvancedModelMonitor(model_path, reference_data_path)