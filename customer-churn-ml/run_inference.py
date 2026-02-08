"""
Main inference script for customer churn prediction.

This script loads a trained model pipeline and generates predictions
on new customer data.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from src.predict import ChurnPredictor
from src.config import MODELS_PATH, RAW_DATA_PATH
from src.utils import setup_logging, validate_model_input


def parse_arguments():
    """Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Run customer churn prediction on new data'
    )
    parser.add_argument(
        '--input',
        type=str,
        default=str(RAW_DATA_PATH / 'test_data.csv'),
        help='Path to input CSV file with customer data'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='predictions.csv',
        help='Path to output CSV file for predictions'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=str(MODELS_PATH / 'churn_prediction_pipeline.joblib'),
        help='Path to trained model pipeline'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    log_level = 'DEBUG' if args.verbose else 'INFO'
    setup_logging(log_level=log_level)
    
    import logging
    logger = logging.getLogger(__name__)
    logger.info("Starting churn prediction inference...")
    
    try:
        # Validate input file exists
        input_path = Path(args.input)
        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            sys.exit(1)
        
        # Load input data
        logger.info(f"Loading data from: {input_path}")
        data = pd.read_csv(input_path)
        logger.info(f"Loaded {len(data)} records")
        
        # Validate input data structure
        validate_model_input(data)
        
        # Initialize predictor
        logger.info(f"Loading model from: {args.model}")
        predictor = ChurnPredictor(model_path=args.model)
        
        # Generate predictions
        logger.info("Generating predictions...")
        predictions = predictor.predict(data)
        probabilities = predictor.predict_proba(data)
        
        # Prepare output dataframe
        output_df = data.copy()
        output_df['churn_prediction'] = predictions
        output_df['churn_probability'] = probabilities[:, 1]  # Probability of churn
        output_df['no_churn_probability'] = probabilities[:, 0]  # Probability of no churn
        
        # Save predictions
        output_path = Path(args.output)
        output_df.to_csv(output_path, index=False)
        logger.info(f"Predictions saved to: {output_path}")
        
        # Summary statistics
        churn_count = predictions.sum()
        churn_rate = (churn_count / len(predictions)) * 100
        logger.info(f"\nPrediction Summary:")
        logger.info(f"  Total customers: {len(predictions)}")
        logger.info(f"  Predicted churners: {churn_count} ({churn_rate:.2f}%)")
        logger.info(f"  Predicted non-churners: {len(predictions) - churn_count} ({100-churn_rate:.2f}%)")
        
        logger.info("\nInference completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
