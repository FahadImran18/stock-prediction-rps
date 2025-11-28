"""
Unit tests for the pipeline
"""
import pytest
import pandas as pd
import numpy as np
from src.data.quality_check import DataQualityChecker
from src.data.transform import StockDataTransformer


def test_quality_checker():
    """Test data quality checker"""
    checker = DataQualityChecker(
        max_null_percentage=1.0,
        required_columns=['timestamp', 'close', 'volume'],
        min_rows=10
    )
    
    # Create test data
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='H'),
        'close': np.random.randn(100),
        'volume': np.random.randint(1000, 10000, 100)
    })
    
    passed, errors = checker.validate(df)
    assert passed == True
    assert len(errors) == 0


def test_quality_checker_fails_on_missing_columns():
    """Test quality checker fails on missing columns"""
    checker = DataQualityChecker(
        required_columns=['timestamp', 'close', 'volume']
    )
    
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='H'),
        'close': np.random.randn(100)
        # Missing 'volume'
    })
    
    passed, errors = checker.validate(df)
    assert passed == False
    assert len(errors) > 0


def test_transformer():
    """Test data transformer"""
    transformer = StockDataTransformer()
    
    # Create test data
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='H'),
        'open': np.random.randn(100) * 100 + 150,
        'high': np.random.randn(100) * 100 + 155,
        'low': np.random.randn(100) * 100 + 145,
        'close': np.random.randn(100) * 100 + 150,
        'volume': np.random.randint(1000, 10000, 100)
    })
    
    df_transformed = transformer.transform(df)
    
    assert len(df_transformed) > 0
    assert 'target_volatility' in df_transformed.columns
    assert 'hour' in df_transformed.columns
    assert 'returns' in df_transformed.columns

