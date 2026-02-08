"""
Demo script to showcase the customer churn ML project functionality.
"""

from src.preprocess import load_raw_data, clean_data, create_preprocessor
from src.utils import get_data_quality_report, setup_logging
from src.config import RAW_DATA_PATH, FEATURE_COLUMNS

def main():
    print("🚀 Customer Churn ML Project Demo")
    print("=" * 50)
    
    # Load and process data
    print("\n📂 Loading sample data...")
    data = load_raw_data('data/raw/test_data.csv')
    print(f"   ✅ Loaded {len(data)} customer records with {len(data.columns)} features")
    
    # Clean data
    print("\n🧹 Cleaning data...")
    cleaned_data = clean_data(data)
    print(f"   ✅ Data cleaning complete. Shape: {cleaned_data.shape}")
    
    # Data quality report
    print("\n📊 Generating data quality report...")
    quality_report = get_data_quality_report(cleaned_data)
    print(f"   📋 Missing values: {sum(quality_report['missing_values'].values())}")
    print(f"   📋 Duplicate rows: {quality_report['duplicated_rows']}")
    print(f"   📋 Numeric columns: {len([col for col, count in quality_report['unique_values_per_column'].items() if cleaned_data[col].dtype in ['int64', 'float64']])}")
    
    # Show preprocessing pipeline
    print("\n🔄 Creating preprocessing pipeline...")
    preprocessor = create_preprocessor()
    print("   ✅ Preprocessing pipeline created with steps:")
    for name, transformer, _ in preprocessor.transformers:
        print(f"      - {name}: {type(transformer).__name__}")
    
    # Show sample data
    print("\n📋 Sample customer data:")
    print(cleaned_data[['customerID', 'tenure', 'MonthlyCharges', 'Contract', 'PaymentMethod']].head(3))
    
    print(f"\n🎯 Project is ready for model training and inference!")
    print("   Next steps:")
    print("   1. Train a model and save to models/churn_prediction_pipeline.joblib")
    print("   2. Use run_inference.py for predictions")
    print("   3. Add notebooks to notebooks/ directory for analysis")

if __name__ == "__main__":
    main()