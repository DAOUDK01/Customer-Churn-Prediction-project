"""
Enterprise Batch Prediction Pipeline
Production-grade batch processing system for large-scale customer portfolio analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
import logging
import json
import joblib
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import argparse
import sys
import os

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from config import PROJECT_ROOT, MODEL_REGISTRY, BATCH_CONFIG
from predict import ChurnPredictor
from advanced_monitoring import AdvancedModelMonitor
from validation import DataValidator

logger = logging.getLogger(__name__)

class BatchPredictionPipeline:
    """
    Enterprise batch prediction pipeline for processing large customer datasets
    with monitoring, validation, and business impact analysis
    """
    
    def __init__(self, model_path: str = None, output_dir: str = None):
        self.model_path = model_path or str(Path(MODEL_REGISTRY) / "simple_churn_model.joblib")
        self.output_dir = Path(output_dir or PROJECT_ROOT / "batch_outputs")
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.predictor = None
        self.monitor = None
        self.validator = DataValidator()
        
        # Processing configuration
        self.batch_size = BATCH_CONFIG.get('batch_size', 1000)
        self.max_workers = BATCH_CONFIG.get('max_workers', 4)
        self.validation_enabled = True
        self.monitoring_enabled = True
        
        # Statistics tracking
        self.processing_stats = {
            'total_processed': 0,
            'total_errors': 0,
            'processing_time': 0.0,
            'start_time': None,
            'end_time': None
        }
    
    def initialize_pipeline(self) -> bool:
        """Initialize the batch prediction pipeline"""
        try:
            logger.info("🚀 Initializing Batch Prediction Pipeline...")
            
            # Load prediction model
            self.predictor = ChurnPredictor(self.model_path)
            load_result = self.predictor.load_model()
            
            if not load_result['success']:
                logger.error(f"❌ Failed to load model: {load_result['error']}")
                return False
            
            logger.info(f"✅ Model loaded: {load_result['model_type']}")
            
            # Initialize monitoring
            if self.monitoring_enabled:
                reference_data_path = PROJECT_ROOT / "data" / "raw" / "test_data.csv"
                self.monitor = AdvancedModelMonitor(
                    model_path=self.model_path,
                    reference_data_path=str(reference_data_path) if reference_data_path.exists() else None
                )
                logger.info("✅ Monitoring system initialized")
            
            logger.info("🎯 Pipeline initialization complete")
            return True
            
        except Exception as e:
            logger.error(f"❌ Pipeline initialization failed: {e}")
            return False
    
    def process_csv_file(self, input_file: str, output_file: str = None, 
                        include_business_impact: bool = True) -> Dict[str, Any]:
        """
        Process a CSV file for batch predictions
        """
        start_time = time.time()
        self.processing_stats['start_time'] = datetime.now()
        
        try:
            logger.info(f"📊 Processing file: {input_file}")
            
            # Load input data
            input_data = pd.read_csv(input_file)
            logger.info(f"✅ Loaded {len(input_data)} records")
            
            # Validate input data
            if self.validation_enabled:
                validation_result = self._validate_input_data(input_data)
                if not validation_result['valid']:
                    logger.error(f"❌ Data validation failed: {validation_result['issues']}")
                    return self._create_error_result("Data validation failed", validation_result['issues'])
            
            # Process in batches
            results = self._process_data_in_batches(input_data)
            
            # Generate output filename if not provided
            if output_file is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = str(self.output_dir / f"batch_predictions_{timestamp}.csv")
            
            # Save results
            self._save_results(results, output_file, include_business_impact)
            
            # Generate business report
            business_report = self._generate_business_report(results, input_data)
            
            # Finalize processing stats
            self.processing_stats['end_time'] = datetime.now()
            self.processing_stats['processing_time'] = time.time() - start_time
            
            return {
                'success': True,
                'input_file': input_file,
                'output_file': output_file,
                'records_processed': len(results),
                'processing_time_seconds': self.processing_stats['processing_time'],
                'business_impact': business_report,
                'statistics': self.processing_stats
            }
            
        except Exception as e:
            logger.error(f"❌ Batch processing failed: {e}")
            return self._create_error_result("Batch processing failed", str(e))
    
    def _validate_input_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate input data before processing"""
        try:
            required_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                return {
                    'valid': False,
                    'issues': f"Missing required columns: {missing_columns}"
                }
            
            # Check for data quality issues
            issues = []
            
            # Check for negative values
            for col in required_columns:
                if (data[col] < 0).any():
                    issues.append(f"Negative values found in {col}")
            
            # Check for unrealistic values
            if (data['tenure'] > 100).any():
                issues.append("Tenure values > 100 months found")
            
            if (data['MonthlyCharges'] > 1000).any():
                issues.append("Monthly charges > $1000 found")
            
            # Check for missing values
            missing_ratio = data[required_columns].isnull().sum().sum() / (len(data) * len(required_columns))
            if missing_ratio > 0.1:
                issues.append(f"High missing value ratio: {missing_ratio:.1%}")
            
            return {
                'valid': len(issues) == 0,
                'issues': issues if issues else None,
                'quality_score': 1.0 - missing_ratio
            }
            
        except Exception as e:
            return {
                'valid': False,
                'issues': f"Validation error: {e}"
            }
    
    def _process_data_in_batches(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process data in batches with parallel processing"""
        logger.info(f"🔄 Processing {len(data)} records in batches of {self.batch_size}")
        
        all_results = []
        batch_count = 0
        
        # Split data into batches
        for i in range(0, len(data), self.batch_size):
            batch = data.iloc[i:i + self.batch_size].copy()
            batch_count += 1
            
            logger.info(f"📦 Processing batch {batch_count} ({len(batch)} records)")
            
            # Process batch
            batch_results = self._process_single_batch(batch, batch_count)
            all_results.append(batch_results)
            
            # Monitor batch if enabled
            if self.monitoring_enabled and self.monitor:
                self._monitor_batch(batch, batch_results['churn_probability'].values)
        
        # Combine all results
        final_results = pd.concat(all_results, ignore_index=True)
        logger.info(f"✅ Completed processing {len(final_results)} records")
        
        return final_results
    
    def _process_single_batch(self, batch: pd.DataFrame, batch_number: int) -> pd.DataFrame:
        """Process a single batch of data"""
        try:
            batch_results = batch.copy()
            
            # Prepare features for prediction
            required_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
            X_batch = batch[required_features].fillna(0)
            
            # Make predictions
            predictions = []
            confidences = []
            risk_labels = []
            
            for idx, row in X_batch.iterrows():
                try:
                    prediction_result = self.predictor.predict_single(
                        tenure=row['tenure'],
                        MonthlyCharges=row['MonthlyCharges'],
                        TotalCharges=row['TotalCharges']
                    )
                    
                    predictions.append(prediction_result['prediction_proba'])
                    confidences.append(prediction_result.get('confidence', 0.85))
                    risk_labels.append(prediction_result['prediction_label'])
                    
                except Exception as e:
                    logger.warning(f"⚠️ Prediction failed for row {idx}: {e}")
                    predictions.append(0.0)
                    confidences.append(0.0)
                    risk_labels.append('ERROR')
                    self.processing_stats['total_errors'] += 1
            
            # Add predictions to results
            batch_results['churn_probability'] = predictions
            batch_results['confidence_score'] = confidences
            batch_results['risk_level'] = risk_labels
            batch_results['prediction_timestamp'] = datetime.now().isoformat()
            batch_results['batch_number'] = batch_number
            
            # Update processing stats
            self.processing_stats['total_processed'] += len(batch_results)
            
            return batch_results
            
        except Exception as e:
            logger.error(f"❌ Batch processing failed for batch {batch_number}: {e}")
            # Return batch with error indicators
            batch_results = batch.copy()
            batch_results['churn_probability'] = 0.0
            batch_results['confidence_score'] = 0.0
            batch_results['risk_level'] = 'ERROR'
            batch_results['error_message'] = str(e)
            
            return batch_results
    
    def _monitor_batch(self, batch_data: pd.DataFrame, predictions: np.ndarray):
        """Monitor batch for drift and performance issues"""
        try:
            monitoring_result = self.monitor.monitor_prediction_batch(
                features=batch_data[['tenure', 'MonthlyCharges', 'TotalCharges']],
                predictions=predictions
            )
            
            # Log monitoring results
            if monitoring_result.get('drift_detected'):
                logger.warning(f"⚠️ Data drift detected in batch")
            
            if monitoring_result.get('alerts'):
                for alert in monitoring_result['alerts']:
                    logger.warning(f"🚨 Alert: {alert['message']}")
                    
        except Exception as e:
            logger.error(f"❌ Batch monitoring failed: {e}")
    
    def _save_results(self, results: pd.DataFrame, output_file: str, include_business_impact: bool):
        """Save prediction results to file"""
        try:
            # Calculate business impact columns
            if include_business_impact:
                results['monthly_revenue_at_risk'] = results.apply(
                    lambda row: row['MonthlyCharges'] if row['risk_level'] == 'HIGH' else 0, axis=1
                )
                
                results['intervention_recommended'] = results['risk_level'].isin(['HIGH', 'MEDIUM'])
                
                results['estimated_ltv'] = results['MonthlyCharges'] * 24  # 24 month LTV
                
                results['priority_score'] = results.apply(
                    lambda row: (
                        3 if row['risk_level'] == 'HIGH' else
                        2 if row['risk_level'] == 'MEDIUM' else 1
                    ), axis=1
                )
            
            # Save to CSV
            results.to_csv(output_file, index=False)
            logger.info(f"💾 Results saved to: {output_file}")
            
            # Save summary statistics
            summary_file = output_file.replace('.csv', '_summary.json')
            summary_stats = self._generate_summary_statistics(results)
            
            with open(summary_file, 'w') as f:
                json.dump(summary_stats, f, indent=2, default=str)
            
            logger.info(f"📊 Summary saved to: {summary_file}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save results: {e}")
            raise
    
    def _generate_summary_statistics(self, results: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary statistics for the batch processing"""
        try:
            total_records = len(results)
            high_risk = len(results[results['risk_level'] == 'HIGH'])
            medium_risk = len(results[results['risk_level'] == 'MEDIUM'])
            low_risk = len(results[results['risk_level'] == 'LOW'])
            errors = len(results[results['risk_level'] == 'ERROR'])
            
            return {
                'processing_summary': {
                    'total_records': total_records,
                    'successful_predictions': total_records - errors,
                    'errors': errors,
                    'processing_time_seconds': self.processing_stats['processing_time'],
                    'records_per_second': total_records / self.processing_stats['processing_time'] if self.processing_stats['processing_time'] > 0 else 0
                },
                'risk_distribution': {
                    'high_risk': {'count': high_risk, 'percentage': high_risk / total_records * 100},
                    'medium_risk': {'count': medium_risk, 'percentage': medium_risk / total_records * 100},
                    'low_risk': {'count': low_risk, 'percentage': low_risk / total_records * 100}
                },
                'business_metrics': {
                    'avg_churn_probability': results['churn_probability'].mean(),
                    'total_monthly_revenue_at_risk': results.get('monthly_revenue_at_risk', pd.Series([0])).sum(),
                    'customers_requiring_intervention': high_risk + medium_risk,
                    'estimated_annual_revenue_at_risk': results.get('monthly_revenue_at_risk', pd.Series([0])).sum() * 12
                },
                'model_info': {
                    'model_version': self.predictor.get_model_info().get('version', '1.0'),
                    'prediction_timestamp': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Summary generation failed: {e}")
            return {'error': str(e)}
    
    def _generate_business_report(self, results: pd.DataFrame, original_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive business impact report"""
        try:
            # Risk analysis
            high_risk_customers = results[results['risk_level'] == 'HIGH']
            medium_risk_customers = results[results['risk_level'] == 'MEDIUM']
            
            # Financial impact
            total_monthly_revenue = results['MonthlyCharges'].sum()
            monthly_revenue_at_risk = high_risk_customers['MonthlyCharges'].sum()
            
            # Customer lifetime value analysis
            avg_monthly_revenue = results['MonthlyCharges'].mean()
            high_risk_ltv_at_risk = len(high_risk_customers) * avg_monthly_revenue * 24
            
            # Intervention planning
            intervention_cost_per_customer = 15.0
            total_intervention_cost = len(high_risk_customers) * intervention_cost_per_customer
            retention_success_rate = 0.35
            expected_customers_saved = len(high_risk_customers) * retention_success_rate
            estimated_revenue_saved = expected_customers_saved * avg_monthly_revenue * 24
            
            # ROI calculation
            net_benefit = estimated_revenue_saved - total_intervention_cost
            roi_percentage = (net_benefit / total_intervention_cost * 100) if total_intervention_cost > 0 else 0
            
            return {
                'executive_summary': {
                    'total_customers_analyzed': len(results),
                    'high_risk_customers': len(high_risk_customers),
                    'medium_risk_customers': len(medium_risk_customers),
                    'overall_churn_risk_rate': len(high_risk_customers) / len(results) * 100
                },
                'financial_impact': {
                    'total_monthly_revenue': total_monthly_revenue,
                    'monthly_revenue_at_risk': monthly_revenue_at_risk,
                    'annual_revenue_at_risk': monthly_revenue_at_risk * 12,
                    'customer_lifetime_value_at_risk': high_risk_ltv_at_risk
                },
                'intervention_analysis': {
                    'recommended_interventions': len(high_risk_customers) + len(medium_risk_customers),
                    'estimated_intervention_cost': total_intervention_cost,
                    'expected_customers_retained': expected_customers_saved,
                    'estimated_revenue_saved': estimated_revenue_saved,
                    'net_business_benefit': net_benefit,
                    'roi_percentage': roi_percentage
                },
                'action_items': [
                    f"Immediately contact {len(high_risk_customers)} high-risk customers",
                    f"Deploy retention campaigns for {len(medium_risk_customers)} medium-risk customers",
                    f"Allocate ${total_intervention_cost:,.0f} for intervention programs",
                    f"Expected ROI: {roi_percentage:.0f}% (${net_benefit:,.0f} net benefit)"
                ]
            }
            
        except Exception as e:
            logger.error(f"❌ Business report generation failed: {e}")
            return {'error': str(e)}
    
    def _create_error_result(self, message: str, details: Any = None) -> Dict[str, Any]:
        """Create error result dictionary"""
        return {
            'success': False,
            'error': message,
            'details': details,
            'timestamp': datetime.now().isoformat()
        }

def main():
    """CLI interface for batch prediction pipeline"""
    parser = argparse.ArgumentParser(description="Enterprise Batch Prediction Pipeline")
    parser.add_argument('input_file', help='Input CSV file path')
    parser.add_argument('--output', '-o', help='Output file path (optional)')
    parser.add_argument('--model', '-m', help='Model file path (optional)')
    parser.add_argument('--batch-size', type=int, default=1000, help='Batch size for processing')
    parser.add_argument('--no-monitoring', action='store_true', help='Disable monitoring')
    parser.add_argument('--no-validation', action='store_true', help='Disable input validation')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Initialize pipeline
    pipeline = BatchPredictionPipeline(
        model_path=args.model,
        output_dir=None
    )
    
    # Configure pipeline
    pipeline.batch_size = args.batch_size
    pipeline.monitoring_enabled = not args.no_monitoring
    pipeline.validation_enabled = not args.no_validation
    
    # Initialize
    if not pipeline.initialize_pipeline():
        logger.error("❌ Pipeline initialization failed")
        return 1
    
    # Process file
    result = pipeline.process_csv_file(
        input_file=args.input_file,
        output_file=args.output
    )
    
    if result['success']:
        logger.info("🎉 Batch processing completed successfully!")
        logger.info(f"📊 Processed {result['records_processed']} records in {result['processing_time_seconds']:.2f} seconds")
        logger.info(f"💾 Output saved to: {result['output_file']}")
        return 0
    else:
        logger.error(f"❌ Batch processing failed: {result['error']}")
        return 1

if __name__ == "__main__":
    sys.exit(main())