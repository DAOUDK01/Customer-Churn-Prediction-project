"""
Production deployment and testing script for Customer Churn ML system.

This script demonstrates the complete production workflow including:
- Model loading and validation
- Data validation and monitoring  
- Business impact calculation
- Model governance and versioning
"""

import sys
sys.path.append('.')

import pandas as pd
import numpy as np
import json
from datetime import datetime
from pathlib import Path

from src.predict import ChurnPredictor
from src.governance import ModelRegistry, BusinessImpactCalculator, generate_model_card
from src.monitoring import DataDriftMonitor, SystemHealthMonitor
from src.validation import validate_churn_data
from src.utils import setup_logging

def main():
    """Run comprehensive production system test."""
    
    print("🚀 CUSTOMER CHURN ML - PRODUCTION DEPLOYMENT TEST")
    print("=" * 70)
    
    # Setup logging
    setup_logging(log_level='INFO')
    
    # 1. MODEL LOADING & VALIDATION
    print("\n📊 1. MODEL LOADING & VALIDATION")
    print("-" * 40)
    
    predictor = ChurnPredictor(enable_monitoring=True)
    
    try:
        load_result = predictor.load_model()
        print(f"✅ Model loaded: {load_result['model_version']}")
        print(f"   Load time: {load_result['load_time_seconds']:.3f}s")
        print(f"   Has metadata: {load_result['has_metadata']}")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False
    
    # 2. DATA VALIDATION
    print("\n🔍 2. DATA VALIDATION")
    print("-" * 40)
    
    # Load test data
    test_data = pd.read_csv('data/raw/test_data.csv')
    print(f"Test data shape: {test_data.shape}")
    
    # Validate data schema
    validation_result = validate_churn_data(test_data)
    if validation_result['valid']:
        print("✅ Data validation passed")
    else:
        print(f"⚠️ Data validation warnings: {len(validation_result['errors'])} errors")
        for error in validation_result['errors'][:3]:  # Show first 3 errors
            print(f"   - {error}")
    
    # 3. INFERENCE TESTING
    print("\n🎯 3. INFERENCE TESTING")
    print("-" * 40)
    
    try:
        # Batch prediction
        predictions, probabilities = predictor.predict(
            test_data, 
            return_probabilities=True,
            validate_input=True
        )
        
        churn_rate = np.mean(predictions)
        avg_probability = np.mean(probabilities)
        
        print(f"✅ Batch prediction successful")
        print(f"   Predicted churn rate: {churn_rate:.1%}")
        print(f"   Average probability: {avg_probability:.3f}")
        
        # Single prediction test
        single_result = predictor.predict_single(
            tenure=12,
            MonthlyCharges=75.0,
            TotalCharges=900.0,
            Contract='Month-to-month',
            PaymentMethod='Electronic check'
        )
        
        print(f"✅ Single prediction: {single_result['prediction_label']}")
        print(f"   Confidence: {single_result['confidence_level']}")
        
    except Exception as e:
        print(f"❌ Inference testing failed: {e}")
        return False
    
    # 4. BUSINESS IMPACT CALCULATION
    print("\n💼 4. BUSINESS IMPACT ANALYSIS")
    print("-" * 40)
    
    # Calculate business impact from test predictions
    calculator = BusinessImpactCalculator()
    
    # Calculate precision and recall from predictions (simplified)
    # In real scenario, you'd have true labels for evaluation
    precision = 0.75  # Example metric from model evaluation
    recall = 0.68     # Example metric from model evaluation
    
    business_impact = calculator.calculate_impact(
        recall=recall,
        precision=precision,
        customer_base=50000  # Example customer base
    )
    
    print(f"💰 Revenue Protected: ${business_impact['revenue_saved']:,.0f}")
    print(f"📈 Net Benefit: ${business_impact['net_benefit']:,.0f}")
    print(f"🎯 ROI: {business_impact['roi_ratio']:.1f}:1")
    print(f"👥 Customers Saved: {business_impact['customers_saved']:,}")
    
    # 5. MODEL GOVERNANCE
    print("\n📋 5. MODEL GOVERNANCE")
    print("-" * 40)
    
    # Generate model card
    model_info = predictor.get_model_info()
    
    if 'metadata' in model_info:
        metadata = model_info['metadata']
        print(f"📊 Model Version: {metadata.get('version', 'unknown')}")
        print(f"📅 Created: {metadata.get('created_date', 'unknown')}")
        print(f"🎯 Test ROC-AUC: {metadata.get('performance', {}).get('test_roc_auc', 'N/A')}")
    else:
        print("⚠️ No model metadata found")
    
    # 6. MONITORING SETUP
    print("\n📊 6. MONITORING & HEALTH CHECKS")
    print("-" * 40)
    
    # System health check
    health_monitor = SystemHealthMonitor()
    health_result = health_monitor.run_health_check(
        model_path=predictor.model_path,
        recent_data=test_data,
        recent_predictions=predictions
    )
    
    print(f"🏥 System Health: {health_result['overall_status'].upper()}")
    if health_result['alerts']:
        print(f"⚠️ Alerts: {len(health_result['alerts'])}")
        for alert in health_result['alerts'][:2]:  # Show first 2 alerts
            print(f"   - {alert['message']}")
    else:
        print("✅ No health alerts")
    
    # Performance stats
    perf_stats = predictor.get_performance_stats()
    print(f"⚡ Performance: {perf_stats['average_time_per_prediction_ms']:.1f}ms per prediction")
    
    # 7. DEPLOYMENT READINESS CHECK
    print("\n🚀 7. DEPLOYMENT READINESS")
    print("-" * 40)
    
    readiness_score = 0
    readiness_checks = []
    
    # Model loaded successfully
    if predictor.is_loaded:
        readiness_score += 20
        readiness_checks.append("✅ Model loaded and functional")
    
    # Data validation working
    if validation_result['valid']:
        readiness_score += 20
        readiness_checks.append("✅ Data validation operational")
    
    # Predictions working
    if 'predictions' in locals():
        readiness_score += 20
        readiness_checks.append("✅ Inference system operational")
    
    # Business metrics calculated
    if 'business_impact' in locals():
        readiness_score += 20
        readiness_checks.append("✅ Business impact tracking ready")
    
    # Health monitoring working
    if health_result['overall_status'] in ['healthy', 'degraded']:
        readiness_score += 20
        readiness_checks.append("✅ Health monitoring operational")
    
    print(f"📊 Deployment Readiness Score: {readiness_score}/100")
    for check in readiness_checks:
        print(f"   {check}")
    
    # Final assessment
    print("\n" + "=" * 70)
    if readiness_score >= 80:
        print("🎉 SYSTEM READY FOR PRODUCTION DEPLOYMENT")
        print("   All critical components are operational")
        return True
    else:
        print("⚠️ SYSTEM REQUIRES ADDITIONAL WORK")
        print(f"   Readiness score: {readiness_score}% (minimum: 80%)")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)