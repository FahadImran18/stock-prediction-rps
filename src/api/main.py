"""
FastAPI application for stock volatility prediction
Includes Prometheus metrics for monitoring
"""
import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from prometheus_fastapi_instrumentator import Instrumentator
import joblib

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.models import PredictionRequest, PredictionResponse, HealthResponse
from src.monitoring.metrics import DataDriftDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Stock Volatility Prediction API",
    description="Real-time stock volatility prediction service",
    version="1.0.0"
)

# Prometheus metrics
request_count = Counter(
    'api_requests_total',
    'Total number of API requests',
    ['method', 'endpoint']
)

request_latency = Histogram(
    'api_request_duration_seconds',
    'API request latency in seconds',
    ['method', 'endpoint']
)

prediction_latency = Histogram(
    'model_inference_duration_seconds',
    'Model inference latency in seconds'
)

data_drift_ratio = Gauge(
    'data_drift_ratio',
    'Ratio of predictions with out-of-distribution features'
)

# Instrument FastAPI with Prometheus
Instrumentator().instrument(app).expose(app)

# Global model and feature columns
model = None
feature_columns = None
drift_detector = None


def load_model():
    """Load the latest model from MLflow"""
    global model, feature_columns, drift_detector
    
    try:
        mlflow_tracking_uri = os.getenv('MLFLOW_TRACKING_URI')
        if not mlflow_tracking_uri:
            # Try to load from local file
            model_path = Path("models/model.joblib")
            if model_path.exists():
                logger.info(f"Loading model from {model_path}")
                model = joblib.load(model_path)
                
                # Load feature columns
                features_path = Path("models/features.pkl")
                if features_path.exists():
                    feature_columns = joblib.load(features_path)
                else:
                    logger.warning("Features file not found, using default")
                    feature_columns = []
                
                drift_detector = DataDriftDetector()
                logger.info("Model loaded successfully from local file")
                return
            else:
                raise ValueError("MLFLOW_TRACKING_URI not set and local model not found")
        
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        logger.info(f"Connecting to MLflow: {mlflow_tracking_uri}")
        
        # Get the latest model from registry
        client = mlflow.tracking.MlflowClient()
        
        try:
            latest_version = client.get_latest_versions(
                "StockVolatilityPredictor",
                stages=["Production"]
            )
            
            if latest_version:
                model_uri = f"models:/StockVolatilityPredictor/Production"
            else:
                # Fallback to latest version
                latest_version = client.get_latest_versions(
                    "StockVolatilityPredictor"
                )
                if latest_version:
                    model_uri = f"models:/StockVolatilityPredictor/{latest_version[0].version}"
                else:
                    raise ValueError("No model found in registry")
        except Exception as e:
            logger.warning(f"Could not get model from registry: {e}")
            # Try to load from local file
            model_path = Path("models/model.joblib")
            if model_path.exists():
                model = joblib.load(model_path)
                features_path = Path("models/features.pkl")
                if features_path.exists():
                    feature_columns = joblib.load(features_path)
                drift_detector = DataDriftDetector()
                logger.info("Model loaded from local file")
                return
            raise
        
        logger.info(f"Loading model from {model_uri}")
        model = mlflow.sklearn.load_model(model_uri)
        
        # Try to load feature columns from artifact
        try:
            run_id = latest_version[0].run_id
            artifacts_path = mlflow.artifacts.download_artifacts(
                run_id=run_id,
                artifact_path="artifacts/features.pkl"
            )
            feature_columns = joblib.load(artifacts_path)
        except Exception as e:
            logger.warning(f"Could not load feature columns: {e}")
            feature_columns = []
        
        drift_detector = DataDriftDetector()
        logger.info("Model loaded successfully from MLflow")
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    logger.info("Starting up API service...")
    load_model()
    logger.info("API service ready")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        model_loaded=model is not None
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict stock volatility
    
    Args:
        request: Prediction request with symbol and optional features
        
    Returns:
        Prediction response with predicted volatility
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Start latency timer
    import time
    start_time = time.time()
    
    try:
        # Prepare features
        if request.features:
            # Use provided features
            feature_dict = request.features
        else:
            # Generate default features (in production, fetch from API)
            feature_dict = {}
        
        # Convert to DataFrame with proper feature order
        if feature_columns:
            # Create DataFrame with all features
            feature_df = pd.DataFrame([feature_dict])
            
            # Ensure all feature columns are present
            for col in feature_columns:
                if col not in feature_df.columns:
                    feature_df[col] = 0  # Default value
            
            # Select only the required features in the correct order
            X = feature_df[feature_columns]
        else:
            # Fallback: use provided features as-is
            X = pd.DataFrame([feature_dict])
        
        # Check for data drift
        if drift_detector:
            is_drift = drift_detector.detect_drift(X)
            if is_drift:
                data_drift_ratio.inc()
        
        # Make prediction
        prediction_start = time.time()
        prediction = model.predict(X)[0]
        prediction_time = time.time() - prediction_start
        
        # Record metrics
        prediction_latency.observe(prediction_time)
        request_latency.labels(method="POST", endpoint="/predict").observe(
            time.time() - start_time
        )
        request_count.labels(method="POST", endpoint="/predict").inc()
        
        return PredictionResponse(
            symbol=request.symbol,
            predicted_volatility=float(prediction),
            timestamp=datetime.now().isoformat(),
            model_version="latest"
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        request_count.labels(method="POST", endpoint="/predict").inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

