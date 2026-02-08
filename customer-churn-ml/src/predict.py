"""
Model prediction utilities for Customer Churn ML project.

Contains functions for loading trained models and making predictions.
"""

import joblib
import numpy as np
import pandas as pd
import logging
from typing import Tuple, Union, Optional
from pathlib import Path

from .config import MODEL_PIPELINE_PATH
from .preprocess import prepare_features_target

logger = logging.getLogger(__name__)


class ChurnPredictor:
    """
    Wrapper class for churn prediction model pipeline.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to the trained model pipeline. If None, uses default path.
        """
        self.model_path = Path(model_path) if model_path else MODEL_PIPELINE_PATH
        self.pipeline = None
        self.is_loaded = False
        
    def load_model(self) -> None:
        """
        Load the trained model pipeline from disk.
        
        Raises:
            FileNotFoundError: If model file doesn't exist.
            Exception: If model loading fails.
        """
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        try:
            self.pipeline = joblib.load(self.model_path)
            self.is_loaded = True
            logger.info(f"Model loaded successfully from: {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Make predictions on input data.
        
        Args:
            X: Input features as DataFrame or array.
            
        Returns:
            Array of predicted labels.
            
        Raises:
            ValueError: If model is not loaded.
        """
        if not self.is_loaded:
            self.load_model()
        
        try:
            predictions = self.pipeline.predict(X)
            logger.info(f"Generated predictions for {len(predictions)} samples")
            return predictions
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict class probabilities for input data.
        
        Args:
            X: Input features as DataFrame or array.
            
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
            probabilities = self.pipeline.predict_proba(X)
            logger.info(f"Generated probabilities for {len(probabilities)} samples")
            return probabilities
        except Exception as e:
            logger.error(f"Probability prediction failed: {str(e)}")
            raise
    
    def predict_single(self, **kwargs) -> Tuple[int, float]:
        """
        Make prediction for a single customer with named features.
        
        Args:
            **kwargs: Feature values as keyword arguments.
            
        Returns:
            Tuple of (predicted_label, churn_probability)
            
        Example:
            prediction, probability = predictor.predict_single(
                tenure=24,
                MonthlyCharges=65.50,
                Contract='Month-to-month'
            )
        """
        # Create DataFrame from input arguments
        input_df = pd.DataFrame([kwargs])
        
        # Make prediction
        prediction = self.predict(input_df)[0]
        
        # Get probability if available
        try:
            proba = self.predict_proba(input_df)[0]
            # Assuming binary classification, get probability of positive class
            churn_probability = proba[1] if len(proba) == 2 else proba.max()
        except ValueError:
            churn_probability = 0.0
        
        logger.info(f"Single prediction: {prediction}, probability: {churn_probability:.3f}")
        return int(prediction), float(churn_probability)


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