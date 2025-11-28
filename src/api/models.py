"""
Pydantic models for API requests and responses
"""
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class PredictionRequest(BaseModel):
    """Request model for prediction"""
    symbol: str = Field(..., description="Stock symbol (e.g., 'AAPL')")
    features: Optional[dict] = Field(None, description="Feature values for prediction")
    timestamp: Optional[str] = Field(None, description="Timestamp for prediction")


class PredictionResponse(BaseModel):
    """Response model for prediction"""
    symbol: str
    predicted_volatility: float = Field(..., description="Predicted volatility")
    timestamp: str = Field(..., description="Prediction timestamp")
    model_version: Optional[str] = Field(None, description="Model version used")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    model_loaded: bool

