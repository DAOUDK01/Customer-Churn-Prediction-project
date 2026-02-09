"""
Production-grade model prediction utilities for Customer Churn ML project.

Contains functions for loading trained models, making predictions,
and monitoring inference in production environments.
"""

import joblib
import numpy as np
import pandas as pd
import logging
from typing import Tuple, Union, Optional, Dict, Any
from pathlib import Path
import time
import json
from datetime import datetime

try:
    from .config import MODEL_PIPELINE_PATH, REPORTS_PATH
except ImportError:
    from config import MODEL_PIPELINE_PATH, REPORTS_PATH

try:
    from .preprocess import prepare_features_target
except ImportError:
    from preprocess import prepare_features_target

try:
    from .validation import validate_churn_data, sanitize_churn_data
except ImportError:
    from validation import validate_churn_data, sanitize_churn_data

try:
    from .monitoring import PredictionMonitor
except ImportError:
    from monitoring import PredictionMonitor

logger = logging.getLogger(__name__)


class ChurnPredictor:
    """
    Production-grade wrapper class for churn prediction model pipeline.
    
    Features:
    - Model versioning and metadata tracking
    - Input validation and sanitization
    - Prediction monitoring and logging
    - Graceful error handling
    - Performance monitoring
    """
    
    def __init__(self, model_path: Optional[str] = None, enable_monitoring: bool = True):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to the trained model pipeline. If None, uses default path.
            enable_monitoring: Whether to enable prediction monitoring
        """
        self.model_path = Path(model_path) if model_path else MODEL_PIPELINE_PATH
        self.pipeline = None
        self.model = None
        self.model_metadata = None
        self.is_loaded = False
        self.enable_monitoring = enable_monitoring
        
        # Initialize monitoring
        if self.enable_monitoring:
            self.monitor = PredictionMonitor()
        
        # Performance tracking
        self.prediction_count = 0
        self.total_inference_time = 0
        
    def load_model(self) -> Dict[str, Any]:
        """
        Load the trained model pipeline from disk with validation.
        
        Returns:
            Dictionary with model loading results
            
        Raises:
            FileNotFoundError: If model file doesn't exist.
            Exception: If model loading fails.
        """
        start_time = time.time()
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        try:
            # Load model
            self.pipeline = joblib.load(self.model_path)
            self.model = self.pipeline
            
            # Try to load metadata if available
            metadata_path = self.model_path.parent / "model_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.model_metadata = json.load(f)
            
            self.is_loaded = True
            load_time = time.time() - start_time
            
            logger.info(f"Model loaded successfully from: {self.model_path}")
            logger.info(f"Model load time: {load_time:.3f}s")
            
            # Return loading summary
            result = {
                "status": "success",
                "success": True,
                "model_path": str(self.model_path),
                "load_time_seconds": load_time,
                "model_version": self.model_metadata.get("version") if self.model_metadata else "unknown",
                "has_metadata": self.model_metadata is not None
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray], 
                validate_input: bool = True,
                return_probabilities: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Make predictions on input data with validation and monitoring.
        
        Args:
            X: Input features as DataFrame or array.
            validate_input: Whether to validate input data
            return_probabilities: Whether to return prediction probabilities
            
        Returns:
            Array of predicted labels, optionally with probabilities
            
        Raises:
            ValueError: If model is not loaded or input is invalid.
        """
        start_time = time.time()
        
        if not self.is_loaded:
            load_result = self.load_model()
            logger.info(f"Auto-loaded model: {load_result['model_version']}")
        
        try:
            # Input validation and sanitization
            if validate_input and isinstance(X, pd.DataFrame):
                validation_result = validate_churn_data(X)
                if not validation_result["valid"]:
                    logger.warning(f"Input validation warnings: {validation_result['errors']}")
                    # Sanitize data to fix common issues
                    X = sanitize_churn_data(X)
            
            # Make predictions
            predictions = self.pipeline.predict(X)
            probabilities = None
            
            if return_probabilities:
                if hasattr(self.pipeline, 'predict_proba'):
                    probabilities = self.pipeline.predict_proba(X)
                    # For binary classification, return probability of positive class
                    if probabilities.shape[1] == 2:
                        probabilities = probabilities[:, 1]
                else:
                    logger.warning("Model does not support probability predictions")
                    probabilities = np.zeros(len(predictions))
            
            # Performance tracking
            inference_time = time.time() - start_time
            self.prediction_count += len(predictions)
            self.total_inference_time += inference_time
            
            logger.info(f"Generated predictions for {len(predictions)} samples in {inference_time:.3f}s")
            
            # Monitoring
            if self.enable_monitoring:
                self._log_predictions(predictions, probabilities, X)
            
            # Return results
            if return_probabilities:
                return predictions, probabilities
            else:
                return predictions
                
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray],
                     validate_input: bool = True) -> np.ndarray:
        """
        Predict class probabilities for input data.
        
        Args:
            X: Input features as DataFrame or array.
            validate_input: Whether to validate input data
            
        Returns:
            Array of predicted probabilities.
            
        Raises:
            ValueError: If model is not loaded or doesn't support probabilities.
        """
        if not self.is_loaded:
            self.load_model()
        
        if not hasattr(self.pipeline, 'predict_proba'):
            raise ValueError("Model pipeline does not support probability predictions")
        
        try:
            # Input validation
            if validate_input and isinstance(X, pd.DataFrame):
                validation_result = validate_churn_data(X)
                if not validation_result["valid"]:
                    logger.warning(f"Input validation warnings: {validation_result['errors']}")
                    X = sanitize_churn_data(X)
            
            probabilities = self.pipeline.predict_proba(X)
            logger.info(f"Generated probabilities for {len(probabilities)} samples")
            return probabilities
            
        except Exception as e:
            logger.error(f"Probability prediction failed: {str(e)}")
            raise
    
    def predict_single(self, **kwargs) -> Dict[str, Any]:
        """
        Make prediction for a single customer with comprehensive output.
        
        Args:
            **kwargs: Feature values as keyword arguments.
            
        Returns:
            Dictionary with prediction results and metadata
            
        Example:
            result = predictor.predict_single(
                tenure=24,
                MonthlyCharges=65.50,
                Contract='Month-to-month'
            )
        """
        start_time = time.time()
        
        # Create DataFrame from input arguments
        input_df = pd.DataFrame([kwargs])
        
        try:
            # Make prediction with probabilities
            prediction, probability = self.predict(
                input_df, 
                validate_input=True,
                return_probabilities=True
            )
            
            # Prepare result
            result = {
                "prediction": int(prediction[0]),
                "prediction_label": "Churn" if prediction[0] == 1 else "Retain",
                "churn_probability": float(probability[0]) if probability is not None else None,
                "confidence_level": self._get_confidence_level(probability[0] if probability is not None else 0.5),
                "input_features": kwargs,
                "model_version": self.model_metadata.get("version") if self.model_metadata else "unknown",
                "prediction_timestamp": datetime.now().isoformat(),
                "processing_time_ms": (time.time() - start_time) * 1000
            }
            
            logger.info(f"Single prediction: {result['prediction_label']} "
                       f"(prob: {result['churn_probability']:.3f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Single prediction failed: {str(e)}")
            raise
    
    def _get_confidence_level(self, probability: float) -> str:
        """Get human-readable confidence level."""
        if probability > 0.8 or probability < 0.2:
            return "High"
        elif probability > 0.6 or probability < 0.4:
            return "Medium"
        else:
            return "Low"
    
    def _log_predictions(self, predictions: np.ndarray,
                        probabilities: Optional[np.ndarray],
                        input_data: Optional[pd.DataFrame]) -> None:
        """Log predictions for monitoring."""
        if self.enable_monitoring:
            try:
                self.monitor.log_predictions(
                    predictions=predictions,
                    probabilities=probabilities,
                    input_data=input_data,
                    model_version=self.model_metadata.get("version") if self.model_metadata else None
                )
            except Exception as e:
                logger.warning(f"Failed to log predictions: {str(e)}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get predictor performance statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        avg_time_per_prediction = (
            self.total_inference_time / self.prediction_count 
            if self.prediction_count > 0 else 0
        )
        
        return {
            "total_predictions": self.prediction_count,
            "total_inference_time": self.total_inference_time,
            "average_time_per_prediction_ms": avg_time_per_prediction * 1000,
            "model_loaded": self.is_loaded,
            "model_path": str(self.model_path),
            "monitoring_enabled": self.enable_monitoring
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information.
        
        Returns:
            Dictionary with model metadata and statistics
        """
        if not self.is_loaded:
            self.load_model()
        
        info = {
            "model_type": type(self.pipeline).__name__,
            "model_path": str(self.model_path),
            "is_loaded": self.is_loaded,
            "performance_stats": self.get_performance_stats()
        }
        
        # Add metadata if available
        if self.model_metadata:
            info["metadata"] = self.model_metadata
        
        # Try to get pipeline information
        if hasattr(self.pipeline, 'named_steps'):
            info["pipeline_steps"] = list(self.pipeline.named_steps.keys())
            
            # Get final estimator info
            final_estimator = list(self.pipeline.named_steps.values())[-1]
            info["final_estimator"] = type(final_estimator).__name__
            
            # Get estimator parameters if available
            if hasattr(final_estimator, 'get_params'):
                info["estimator_params"] = final_estimator.get_params()
        
        return info


def load_model_pipeline(model_path: Optional[str] = None):
    """
    Load trained model pipeline from disk.
    
    Args:
        model_path: Path to model file. If None, uses default path.
        
    Returns:
        Loaded model pipeline object.
        
    Raises:
        FileNotFoundError: If model file doesn't exist.
    """
    path = Path(model_path) if model_path else MODEL_PIPELINE_PATH
    
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    
    pipeline = joblib.load(path)
    logger.info(f"Model pipeline loaded from: {path}")
    return pipeline


def predict_churn(data: Union[pd.DataFrame, str], 
                  model_path: Optional[str] = None) -> pd.DataFrame:
    """
    Predict churn for customer data.
    
    Args:
        data: Customer data as DataFrame or path to CSV file.
        model_path: Path to trained model. If None, uses default path.
        
    Returns:
        DataFrame with original data plus predictions.
    """
    # Load data if path provided
    if isinstance(data, str):
        data = pd.read_csv(data)
        logger.info(f"Loaded data from: {data}")
    
    # Initialize predictor
    predictor = ChurnPredictor(model_path)
    
    # Make predictions
    predictions = predictor.predict(data)
    
    # Add predictions to DataFrame
    result_df = data.copy()
    result_df['Predicted_Churn'] = predictions
    
    # Add probabilities if available
    try:
        probabilities = predictor.predict_proba(data)
        if probabilities.shape[1] == 2:  # Binary classification
            result_df['Churn_Probability'] = probabilities[:, 1]
        else:
            result_df['Churn_Probability'] = probabilities.max(axis=1)
    except ValueError:
        logger.info("Probabilities not available for this model")
    
    logger.info(f"Predictions completed for {len(result_df)} customers")
    return result_df


def batch_predict(input_file: str, output_file: str, 
                  model_path: Optional[str] = None) -> None:
    """
    Perform batch prediction on a CSV file and save results.
    
    Args:
        input_file: Path to input CSV file.
        output_file: Path to save predictions.
        model_path: Path to trained model. If None, uses default path.
    """
    logger.info(f"Starting batch prediction: {input_file} -> {output_file}")
    
    # Load data and make predictions
    results = predict_churn(input_file, model_path)
    
    # Save results
    results.to_csv(output_file, index=False)
    logger.info(f"Batch predictions saved to: {output_file}")


def get_feature_importance(model_path: Optional[str] = None) -> pd.DataFrame:
    """
    Extract feature importance from trained model if available.
    
    Args:
        model_path: Path to trained model. If None, uses default path.
        
    Returns:
        DataFrame with feature names and importance values.
    """
    pipeline = load_model_pipeline(model_path)
    
    # Try to extract feature importance
    try:
        # Check if pipeline has a final estimator with feature_importances_
        if hasattr(pipeline, 'named_steps'):
            # Get the last step (classifier)
            classifier = list(pipeline.named_steps.values())[-1]
            if hasattr(classifier, 'feature_importances_'):
                importance_values = classifier.feature_importances_
                
                # Get feature names from preprocessor
                preprocessor = None
                for step_name, step in pipeline.named_steps.items():
                    if hasattr(step, 'transform'):
                        preprocessor = step
                        break
                
                if preprocessor and hasattr(preprocessor, 'get_feature_names_out'):
                    feature_names = preprocessor.get_feature_names_out()
                else:
                    feature_names = [f'feature_{i}' for i in range(len(importance_values))]
                
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importance_values
                }).sort_values('importance', ascending=False)
                
                logger.info(f"Extracted feature importance for {len(importance_df)} features")
                return importance_df
        
        elif hasattr(pipeline, 'feature_importances_'):
            # Direct model without pipeline
            importance_values = pipeline.feature_importances_
            feature_names = [f'feature_{i}' for i in range(len(importance_values))]
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance_values
            }).sort_values('importance', ascending=False)
            
            return importance_df
        
        else:
            logger.warning("Model does not support feature importance extraction")
            return pd.DataFrame()
    
    except Exception as e:
        logger.error(f"Failed to extract feature importance: {str(e)}")
        return pd.DataFrame()


def model_info(model_path: Optional[str] = None) -> dict:
    """
    Get information about the trained model.
    
    Args:
        model_path: Path to trained model. If None, uses default path.
        
    Returns:
        Dictionary with model information.
    """
    pipeline = load_model_pipeline(model_path)
    
    info = {
        'model_type': type(pipeline).__name__,
        'pipeline_steps': []
    }
    
    if hasattr(pipeline, 'named_steps'):
        info['pipeline_steps'] = list(pipeline.named_steps.keys())
        
        # Get final estimator info
        final_estimator = list(pipeline.named_steps.values())[-1]
        info['final_estimator'] = type(final_estimator).__name__
        
        # Get estimator parameters if available
        if hasattr(final_estimator, 'get_params'):
            info['estimator_params'] = final_estimator.get_params()
    
    logger.info(f"Model info extracted: {info['model_type']}")
    return info