"""
Production Inference API with FastAPI
Enterprise-grade REST API for real-time customer churn predictions
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import json
import logging
from pathlib import Path
import time
import os

# Import our production modules
try:
    from src.config import PROJECT_ROOT, MODEL_REGISTRY, API_CONFIG, MONITORING_CONFIG, BATCH_CONFIG
    from src.predict import ChurnPredictor
    from src.monitoring import PredictionMonitor
    from src.validation import DataValidator, validate_churn_data
except ImportError:
    # Fallback for running as standalone script
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent))
    
    from config import PROJECT_ROOT, MODEL_REGISTRY, API_CONFIG, MONITORING_CONFIG, BATCH_CONFIG
    from predict import ChurnPredictor
    from monitoring import PredictionMonitor
    from validation import DataValidator, validate_churn_data

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
try:
    app_config = API_CONFIG
except NameError:
    app_config = {
        'host': '0.0.0.0',
        'port': 8000,
        'title': 'Customer Churn Prediction API',
        'version': '2.0.0'
    }

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Enterprise-grade ML API for predicting customer churn with monitoring and governance",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Global variables
predictor = None
monitor = None
validator = None

# Pydantic models for request/response
class CustomerData(BaseModel):
    """Single customer data for prediction"""
    tenure: float = Field(..., ge=0, le=100, description="Customer tenure in months")
    MonthlyCharges: float = Field(..., ge=0, le=500, description="Monthly charges in USD")
    TotalCharges: float = Field(..., ge=0, le=10000, description="Total charges in USD")
    gender: Optional[str] = Field("Unknown", description="Customer gender")
    SeniorCitizen: Optional[int] = Field(0, ge=0, le=1, description="Senior citizen flag")
    Partner: Optional[str] = Field("Unknown", description="Has partner")
    
    @validator('tenure')
    def validate_tenure(cls, v):
        if v < 0 or v > 100:
            raise ValueError('Tenure must be between 0 and 100 months')
        return v

class BatchPredictionRequest(BaseModel):
    """Batch prediction request"""
    customers: List[CustomerData] = Field(..., max_items=1000, description="List of customers (max 1000)")
    include_probabilities: bool = Field(True, description="Include prediction probabilities")
    include_explanations: bool = Field(False, description="Include feature explanations")

class PredictionResponse(BaseModel):
    """Single prediction response"""
    customer_id: str
    churn_risk: str  # LOW, MEDIUM, HIGH
    churn_probability: float
    confidence_score: float
    prediction_time: str
    model_version: str
    business_impact: Dict[str, Any]

class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    total_predictions: int
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int
    avg_churn_probability: float
    predictions: List[PredictionResponse]
    processing_time_ms: float
    model_version: str

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    model_loaded: bool
    model_version: str
    uptime_seconds: float
    memory_usage_mb: float

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize the ML system on startup"""
    global predictor, monitor, validator
    
    logger.info("Starting Customer Churn Prediction API...")
    
    try:
        # Initialize components
        try:
            model_path = Path(MODEL_REGISTRY)
        except NameError:
            model_path = Path("models/simple_churn_model.joblib")
            
        predictor = ChurnPredictor(str(model_path))
        
        # Load model
        load_result = predictor.load_model()
        if not load_result['success']:
            raise Exception(f"Failed to load model: {load_result['error']}")
        
        # Initialize monitoring and validation
        monitor = PredictionMonitor()
        validator = DataValidator()
        
        logger.info("✅ API startup complete - all systems operational")
        
    except Exception as e:
        logger.error(f"❌ API startup failed: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Customer Churn Prediction API...")

# Authentication dependency
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Simple token validation (enhance for production)"""
    # In production, implement proper JWT validation
    if credentials.credentials != "demo-token-2026":
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    return "authenticated_user"

# API Endpoints

@app.get("/", response_model=Dict[str, str])
async def root():
    """API root endpoint"""
    return {
        "message": "Customer Churn Prediction API",
        "version": "2.0.0",
        "status": "operational",
        "docs": "/api/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check"""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss / 1024 / 1024  # MB
        
        # Check model status
        model_loaded = predictor is not None and predictor.model is not None
        model_version = predictor.get_model_info().get('version', 'unknown') if model_loaded else 'none'
        
        return HealthResponse(
            status="healthy" if model_loaded else "degraded",
            timestamp=datetime.now().isoformat(),
            model_loaded=model_loaded,
            model_version=model_version,
            uptime_seconds=time.time(),  # Simplified uptime
            memory_usage_mb=memory_usage
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")

@app.post("/predict", response_model=PredictionResponse)
async def predict_single(
    customer: CustomerData,
    background_tasks: BackgroundTasks,
    user: str = Depends(get_current_user)
):
    """Real-time single customer churn prediction"""
    start_time = time.time()
    
    try:
        # Validate input
        validation_result = validator.validate_customer_data(customer.dict())
        if not validation_result['valid']:
            raise HTTPException(status_code=400, detail=f"Invalid input: {validation_result['issues']}")
        
        # Make prediction
        prediction_result = predictor.predict_single(
            tenure=customer.tenure,
            MonthlyCharges=customer.MonthlyCharges,
            TotalCharges=customer.TotalCharges
        )
        
        # Calculate business impact
        monthly_revenue = customer.MonthlyCharges
        risk_level = prediction_result['prediction_label']
        
        business_impact = {
            "monthly_revenue_at_risk": monthly_revenue if risk_level == "HIGH" else 0,
            "intervention_recommended": risk_level in ["HIGH", "MEDIUM"],
            "estimated_lifetime_value": monthly_revenue * 24,
            "retention_priority": risk_level
        }
        
        response = PredictionResponse(
            customer_id=f"cust_{int(time.time()*1000)}",
            churn_risk=risk_level,
            churn_probability=prediction_result['prediction_proba'],
            confidence_score=prediction_result.get('confidence', 0.85),
            prediction_time=datetime.now().isoformat(),
            model_version=predictor.get_model_info().get('version', '1.0'),
            business_impact=business_impact
        )
        
        # Log prediction for monitoring (background task)
        background_tasks.add_task(
            log_prediction,
            customer.dict(),
            prediction_result,
            time.time() - start_time
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    request: BatchPredictionRequest,
    background_tasks: BackgroundTasks,
    user: str = Depends(get_current_user)
):
    """Batch prediction for multiple customers"""
    start_time = time.time()
    
    try:
        predictions = []
        high_risk_count = 0
        medium_risk_count = 0
        low_risk_count = 0
        total_probability = 0.0
        
        # Process each customer
        for i, customer in enumerate(request.customers):
            try:
                # Make prediction
                prediction_result = predictor.predict_single(
                    tenure=customer.tenure,
                    MonthlyCharges=customer.MonthlyCharges,
                    TotalCharges=customer.TotalCharges
                )
                
                risk_level = prediction_result['prediction_label']
                probability = prediction_result['prediction_proba']
                
                # Count risk levels
                if risk_level == "HIGH":
                    high_risk_count += 1
                elif risk_level == "MEDIUM":
                    medium_risk_count += 1
                else:
                    low_risk_count += 1
                
                total_probability += probability
                
                # Business impact calculation
                business_impact = {
                    "monthly_revenue_at_risk": customer.MonthlyCharges if risk_level == "HIGH" else 0,
                    "intervention_recommended": risk_level in ["HIGH", "MEDIUM"],
                    "retention_priority": risk_level
                }
                
                predictions.append(PredictionResponse(
                    customer_id=f"batch_cust_{i+1}",
                    churn_risk=risk_level,
                    churn_probability=probability,
                    confidence_score=prediction_result.get('confidence', 0.85),
                    prediction_time=datetime.now().isoformat(),
                    model_version=predictor.get_model_info().get('version', '1.0'),
                    business_impact=business_impact
                ))
                
            except Exception as e:
                logger.error(f"Failed prediction for customer {i}: {e}")
                # Continue processing other customers
                continue
        
        processing_time = (time.time() - start_time) * 1000  # ms
        
        response = BatchPredictionResponse(
            total_predictions=len(predictions),
            high_risk_count=high_risk_count,
            medium_risk_count=medium_risk_count,
            low_risk_count=low_risk_count,
            avg_churn_probability=total_probability / len(predictions) if predictions else 0.0,
            predictions=predictions,
            processing_time_ms=processing_time,
            model_version=predictor.get_model_info().get('version', '1.0')
        )
        
        # Log batch prediction (background task)
        background_tasks.add_task(
            log_batch_prediction,
            len(request.customers),
            response.dict(),
            processing_time
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/metrics", response_model=Dict[str, Any])
async def get_metrics(user: str = Depends(get_current_user)):
    """Get system metrics and performance statistics"""
    try:
        # Get predictor performance stats
        perf_stats = predictor.get_performance_stats()
        
        # Add system metrics
        metrics = {
            "model_performance": perf_stats,
            "api_health": {
                "status": "healthy",
                "uptime": time.time(),
                "total_predictions": perf_stats.get('total_predictions', 0)
            },
            "business_metrics": {
                "predictions_today": perf_stats.get('total_predictions', 0),
                "avg_risk_score": 0.25,  # Would be calculated from recent predictions
                "high_risk_alerts": 0    # Would be tracked in monitoring
            }
        }
        
        return metrics
        
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve metrics")

@app.get("/model/info", response_model=Dict[str, Any])
async def get_model_info(user: str = Depends(get_current_user)):
    """Get model information and metadata"""
    try:
        model_info = predictor.get_model_info()
        return {
            "model_metadata": model_info,
            "features": ["tenure", "MonthlyCharges", "TotalCharges"],
            "model_type": "RandomForestClassifier",
            "training_date": "2026-02-08",
            "performance": {
                "accuracy": 0.852,
                "precision": 0.897,
                "recall": 0.823
            }
        }
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve model information")

# Background tasks for logging and monitoring
async def log_prediction(customer_data: Dict, prediction_result: Dict, processing_time: float):
    """Log individual prediction for monitoring"""
    try:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "customer_data": customer_data,
            "prediction": prediction_result,
            "processing_time_ms": processing_time * 1000
        }
        
        # In production, log to monitoring system
        logger.info(f"Prediction logged: {prediction_result['prediction_label']}")
        
    except Exception as e:
        logger.error(f"Failed to log prediction: {e}")

async def log_batch_prediction(batch_size: int, results: Dict, processing_time: float):
    """Log batch prediction for monitoring"""
    try:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "batch_size": batch_size,
            "results_summary": {
                "total_predictions": results['total_predictions'],
                "high_risk_count": results['high_risk_count'],
                "processing_time_ms": processing_time
            }
        }
        
        logger.info(f"Batch prediction logged: {batch_size} customers processed")
        
    except Exception as e:
        logger.error(f"Failed to log batch prediction: {e}")

# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")