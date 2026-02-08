"""
Streamlit Demo UI for Customer Churn Prediction API

A minimal demonstration interface that consumes the production FastAPI backend.
This UI contains NO ML logic - all predictions come from the API.
"""

import streamlit as st
import requests
import pandas as pd
import json
from typing import Dict, Any, Optional
import time

# Configuration
API_BASE_URL = "http://localhost:8000"
AUTH_TOKEN = "demo-token-2026"

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction Demo",
    page_icon="🎯",
    layout="wide"
)

def check_api_health() -> bool:
    """Check if the API is running and healthy"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def call_single_prediction_api(tenure: float, monthly_charges: float, total_charges: float) -> Optional[Dict[str, Any]]:
    """Call the API for single customer prediction"""
    try:
        headers = {"Authorization": f"Bearer {AUTH_TOKEN}"}
        payload = {
            "tenure": tenure,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges
        }
        
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=payload,
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        st.error(f"Connection Error: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Unexpected Error: {str(e)}")
        return None

def call_batch_prediction_api(customers_data: list) -> Optional[Dict[str, Any]]:
    """Call the API for batch prediction"""
    try:
        headers = {"Authorization": f"Bearer {AUTH_TOKEN}"}
        payload = {
            "customers": customers_data,
            "include_probabilities": True,
            "include_explanations": False
        }
        
        response = requests.post(
            f"{API_BASE_URL}/predict/batch",
            json=payload,
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        st.error(f"Connection Error: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Unexpected Error: {str(e)}")
        return None

def display_single_prediction_result(result: Dict[str, Any]):
    """Display results for single customer prediction"""
    
    # Main prediction result
    col1, col2, col3 = st.columns(3)
    
    with col1:
        risk_color = {
            "HIGH": "🔴",
            "MEDIUM": "🟡", 
            "LOW": "🟢"
        }.get(result['churn_risk'], "⚪")
        
        st.metric(
            label="Churn Risk Level",
            value=f"{risk_color} {result['churn_risk']}"
        )
    
    with col2:
        st.metric(
            label="Churn Probability",
            value=f"{result['churn_probability']:.1%}"
        )
    
    with col3:
        st.metric(
            label="Confidence Score",
            value=f"{result['confidence_score']:.1%}"
        )
    
    # Business impact (if available)
    if 'business_impact' in result:
        st.subheader("📊 Business Impact")
        impact = result['business_impact']
        
        col1, col2 = st.columns(2)
        
        with col1:
            if impact.get('monthly_revenue_at_risk', 0) > 0:
                st.warning(f"💰 Monthly Revenue at Risk: ${impact['monthly_revenue_at_risk']:.2f}")
            else:
                st.info("💰 No immediate revenue risk identified")
        
        with col2:
            if impact.get('intervention_recommended'):
                st.warning("🎯 Intervention Recommended")
            else:
                st.success("✅ Standard Monitoring Sufficient")

def display_batch_prediction_results(result: Dict[str, Any]):
    """Display results for batch predictions"""
    
    # Summary metrics
    st.subheader("📊 Batch Prediction Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Predictions",
            value=result['total_predictions']
        )
    
    with col2:
        st.metric(
            label="🔴 High Risk",
            value=result['high_risk_count']
        )
    
    with col3:
        st.metric(
            label="🟡 Medium Risk", 
            value=result['medium_risk_count']
        )
    
    with col4:
        st.metric(
            label="🟢 Low Risk",
            value=result['low_risk_count']
        )
    
    # Processing info
    st.info(f"⚡ Processing Time: {result['processing_time_ms']:.0f}ms | "
            f"Average Churn Risk: {result['avg_churn_probability']:.1%}")
    
    # Detailed results table
    if result['predictions']:
        st.subheader("📋 Detailed Results")
        
        # Convert to DataFrame for display
        predictions_df = pd.DataFrame([
            {
                'Customer ID': pred['customer_id'],
                'Churn Risk': pred['churn_risk'],
                'Probability': f"{pred['churn_probability']:.1%}",
                'Confidence': f"{pred['confidence_score']:.1%}",
                'Revenue at Risk': f"${pred['business_impact'].get('monthly_revenue_at_risk', 0):.2f}",
                'Action Needed': "Yes" if pred['business_impact'].get('intervention_recommended') else "No"
            }
            for pred in result['predictions'][:50]  # Show first 50 for demo
        ])
        
        st.dataframe(predictions_df, use_container_width=True)
        
        if len(result['predictions']) > 50:
            st.info(f"Showing first 50 of {len(result['predictions'])} predictions")

def main():
    """Main Streamlit application"""
    
    # Header
    st.title("🎯 Customer Churn Prediction Demo")
    st.markdown("*Demonstrating production ML API capabilities*")
    
    # API Health Check
    if not check_api_health():
        st.error("🚨 **API Server Not Running**")
        st.markdown("""
        **To start the API server:**
        ```bash
        cd customer-churn-ml
        python src/api.py
        ```
        Then refresh this page.
        """)
        st.stop()
    else:
        st.success("✅ API Server Connected")
    
    # Mode Selection
    mode = st.sidebar.radio(
        "🔧 Select Prediction Mode",
        ["Single Customer", "Batch Processing"]
    )
    
    if mode == "Single Customer":
        st.header("👤 Single Customer Prediction")
        
        # Input form
        with st.form("single_prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                tenure = st.number_input(
                    "Tenure (months)",
                    min_value=0.0,
                    max_value=100.0,
                    value=24.0,
                    help="How long the customer has been with the company"
                )
                
                monthly_charges = st.number_input(
                    "Monthly Charges ($)",
                    min_value=0.0,
                    max_value=500.0,
                    value=65.0,
                    help="Customer's monthly bill amount"
                )
            
            with col2:
                total_charges = st.number_input(
                    "Total Charges ($)",
                    min_value=0.0,
                    max_value=10000.0,
                    value=1500.0,
                    help="Total amount customer has paid"
                )
                
                # Quick presets
                st.markdown("**Quick Presets:**")
                preset_col1, preset_col2 = st.columns(2)
                
                with preset_col1:
                    if st.form_submit_button("📈 High Value Customer"):
                        tenure, monthly_charges, total_charges = 48.0, 95.0, 4560.0
                
                with preset_col2:
                    if st.form_submit_button("📉 New Customer"):
                        tenure, monthly_charges, total_charges = 3.0, 85.0, 255.0
            
            submitted = st.form_submit_button("🔮 Predict Churn Risk", type="primary")
            
            if submitted:
                with st.spinner("Making prediction..."):
                    result = call_single_prediction_api(tenure, monthly_charges, total_charges)
                    
                    if result:
                        st.success("✅ Prediction Complete!")
                        display_single_prediction_result(result)
    
    else:  # Batch Processing
        st.header("📊 Batch Prediction Processing")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload Customer Data (CSV)",
            type=['csv'],
            help="CSV file with columns: tenure, MonthlyCharges, TotalCharges"
        )
        
        if uploaded_file is not None:
            try:
                # Read and display uploaded data
                df = pd.read_csv(uploaded_file)
                st.subheader("📋 Uploaded Data Preview")
                st.dataframe(df.head(10), use_container_width=True)
                
                # Validate required columns
                required_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if missing_columns:
                    st.error(f"❌ Missing required columns: {', '.join(missing_columns)}")
                    st.info("Required columns: tenure, MonthlyCharges, TotalCharges")
                else:
                    # Process batch
                    if st.button("🚀 Process Batch Predictions", type="primary"):
                        
                        # Limit batch size for demo
                        if len(df) > 100:
                            st.warning(f"⚠️ Processing first 100 rows of {len(df)} for demo purposes")
                            df = df.head(100)
                        
                        # Prepare data for API
                        customers_data = df[required_columns].fillna(0).to_dict('records')
                        
                        with st.spinner(f"Processing {len(customers_data)} customers..."):
                            result = call_batch_prediction_api(customers_data)
                            
                            if result:
                                st.success("✅ Batch Processing Complete!")
                                display_batch_prediction_results(result)
            
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
                st.info("Please ensure your CSV file has the correct format and column names.")
        
        else:
            # Sample data download
            st.info("💡 **Need sample data?** Download the template below:")
            
            sample_data = pd.DataFrame({
                'tenure': [12, 24, 36, 48, 6],
                'MonthlyCharges': [50.0, 65.0, 80.0, 45.0, 95.0],
                'TotalCharges': [600.0, 1560.0, 2880.0, 2160.0, 570.0]
            })
            
            st.download_button(
                label="📥 Download Sample CSV",
                data=sample_data.to_csv(index=False),
                file_name="sample_customers.csv",
                mime="text/csv"
            )
    
    # Footer
    st.markdown("---")
    st.markdown(
        "🔧 **Technical Note:** This UI demonstrates the production API. "
        "All ML logic, monitoring, and business rules remain in the backend API."
    )

if __name__ == "__main__":
    main()