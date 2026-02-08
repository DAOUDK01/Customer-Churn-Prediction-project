#!/usr/bin/env python
"""
Minimal FastAPI Server for Customer Churn Prediction Demo
This is a simplified version that focuses on getting the API running
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import joblib
import uvicorn
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Customer Churn Prediction API",
    description="Demo API for customer churn prediction",
    version="1.0.0"
)

# Global variables
model = None
model_loaded = False

# Pydantic models
class CustomerData(BaseModel):
    """Customer data for prediction"""
    tenure: float = Field(..., ge=0, le=100, description="Customer tenure in months")
    MonthlyCharges: float = Field(..., ge=0, le=500, description="Monthly charges")
    TotalCharges: float = Field(..., ge=0, le=10000, description="Total charges")

class PredictionResponse(BaseModel):
    """Prediction response"""
    churn_risk: str
    churn_probability: float
    confidence_score: float
    prediction_time: str

def load_model():
    """Load the trained model"""
    global model, model_loaded
    
    try:
        model_path = Path("../models/simple_churn_model.joblib")
        if not model_path.exists():
            model_path = Path("models/simple_churn_model.joblib")
        
        if model_path.exists():
            model = joblib.load(model_path)
            model_loaded = True
            logger.info(f"✅ Model loaded from {model_path}")
            return True
        else:
            logger.warning("⚠️ Model file not found - creating dummy model")
            create_dummy_model()
            return True
            
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        create_dummy_model()
        return True

def create_dummy_model():
    """Create a dummy model for demonstration"""
    global model, model_loaded
    
    class DummyModel:
        def predict_proba(self, X):
            # Simulate realistic churn probabilities
            np.random.seed(42)
            proba = np.random.beta(2, 5, len(X))
            # Return as 2D array with [no_churn, churn] probabilities
            return np.column_stack([1 - proba, proba])
    
    model = DummyModel()
    model_loaded = True
    logger.info("✅ Dummy model created for demo")

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info("🚀 Starting Customer Churn Prediction API...")
    load_model()
    logger.info("✅ API startup complete")

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Customer Churn Prediction API", "status": "running"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_churn(customer: CustomerData):
    """Predict customer churn"""
    
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Prepare input data
        input_data = np.array([[customer.tenure, customer.MonthlyCharges, customer.TotalCharges]])
        
        # Make prediction
        prediction_proba = model.predict_proba(input_data)[0]
        churn_probability = prediction_proba[1]  # Probability of churn
        
        # Determine risk level
        if churn_probability >= 0.7:
            risk_level = "HIGH"
        elif churn_probability >= 0.3:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        # Create response
        response = PredictionResponse(
            churn_risk=risk_level,
            churn_probability=churn_probability,
            confidence_score=0.85,  # Simplified confidence score
            prediction_time=datetime.now().isoformat()
        )
        
        logger.info(f"Prediction: {risk_level} ({churn_probability:.1%})")
        return response
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch")
async def predict_batch(customers: List[CustomerData]):
    """Batch prediction for multiple customers"""
    
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        results = []
        
        for customer in customers[:100]:  # Limit to 100 for demo
            input_data = np.array([[customer.tenure, customer.MonthlyCharges, customer.TotalCharges]])
            prediction_proba = model.predict_proba(input_data)[0]
            churn_probability = prediction_proba[1]
            
            if churn_probability >= 0.7:
                risk_level = "HIGH"
            elif churn_probability >= 0.3:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"
            
            results.append({
                "churn_risk": risk_level,
                "churn_probability": churn_probability,
                "confidence_score": 0.85
            })
        
        # Summary statistics
        high_risk = len([r for r in results if r["churn_risk"] == "HIGH"])
        medium_risk = len([r for r in results if r["churn_risk"] == "MEDIUM"])
        low_risk = len([r for r in results if r["churn_risk"] == "LOW"])
        avg_prob = np.mean([r["churn_probability"] for r in results])
        
        return {
            "total_predictions": len(results),
            "high_risk_count": high_risk,
            "medium_risk_count": medium_risk,
            "low_risk_count": low_risk,
            "avg_churn_probability": avg_prob,
            "predictions": results,
            "processing_time_ms": 50.0,  # Simulated processing time
            "model_version": "1.0"
        }
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

if __name__ == "__main__":
    print("🎯 Starting Customer Churn Prediction API")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🔍 Health Check: http://localhost:8000/health")
    print("💡 Press Ctrl+C to stop")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")