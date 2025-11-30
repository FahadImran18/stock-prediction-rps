"""
Monitoring metrics and drift detection
"""
import numpy as np
import pandas as pd
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataDriftDetector:
    """Detect data drift in incoming features"""
    
    def __init__(self, threshold: float = 3.0):
        """
        Initialize drift detector
        
        Args:
            threshold: Z-score threshold for drift detection
        """
        self.threshold = threshold
        self.feature_stats = {}  # Will be populated from training data
    
    def fit(self, training_data: pd.DataFrame):
        """
        Fit detector on training data
        
        Args:
            training_data: Training DataFrame
        """
        for col in training_data.columns:
            if pd.api.types.is_numeric_dtype(training_data[col]):
                self.feature_stats[col] = {
                    'mean': training_data[col].mean(),
                    'std': training_data[col].std()
                }
    
    def detect_drift(self, data: pd.DataFrame) -> bool:
        """
        Detect if data has drifted
        
        Args:
            data: Input DataFrame
            
        Returns:
            True if drift detected
        """
        if not self.feature_stats:
            return False
        
        drift_detected = False
        
        for col in data.columns:
            if col in self.feature_stats and pd.api.types.is_numeric_dtype(data[col]):
                stats = self.feature_stats[col]
                mean = stats['mean']
                std = stats['std']
                
                if std > 0:  # Avoid division by zero
                    z_scores = np.abs((data[col] - mean) / std)
                    if (z_scores > self.threshold).any():
                        drift_detected = True
                        logger.warning(f"Drift detected in feature {col}")
        
        return drift_detected

