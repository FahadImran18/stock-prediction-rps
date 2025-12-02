"""
Data transformation and feature engineering for time-series stock data
"""
import pandas as pd
import numpy as np
from typing import List, Optional
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockDataTransformer:
    """Transform and engineer features for stock volatility prediction"""
    
    def __init__(self):
        self.feature_columns = []
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the raw data
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Cleaning data...")
        df_clean = df.copy()
        
        # Convert timestamp to datetime if it's a string
        if 'timestamp' in df_clean.columns:
            df_clean['timestamp'] = pd.to_datetime(df_clean['timestamp'])
        elif 'date' in df_clean.columns:
            df_clean['timestamp'] = pd.to_datetime(df_clean['date'])
            df_clean = df_clean.drop('date', axis=1)
        
        # Ensure timestamp column exists (should already be set by extract.py)
        if 'timestamp' not in df_clean.columns:
            raise ValueError("DataFrame must have 'timestamp' column")
        
        # Sort by timestamp
        df_clean = df_clean.sort_values('timestamp').reset_index(drop=True)
        
        # Remove duplicates
        df_clean = df_clean.drop_duplicates(subset=['timestamp'], keep='last')
        
        # Ensure numeric columns are numeric
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        logger.info(f"Data cleaned: {len(df_clean)} rows")
        return df_clean
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features
        
        Args:
            df: DataFrame with timestamp column
            
        Returns:
            DataFrame with time features
        """
        logger.info("Creating time features...")
        df = df.copy()
        
        if 'timestamp' not in df.columns:
            raise ValueError("DataFrame must have 'timestamp' column")
        
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Cyclical encoding for time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df
    
    def create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create price-based features
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            DataFrame with price features
        """
        logger.info("Creating price features...")
        df = df.copy()
        
        if 'close' not in df.columns:
            raise ValueError("DataFrame must have 'close' column")
        
        # Price changes
        df['price_change'] = df['close'].diff()
        df['price_change_pct'] = df['close'].pct_change()
        
        # High-Low spread
        if 'high' in df.columns and 'low' in df.columns:
            df['hl_spread'] = df['high'] - df['low']
            df['hl_spread_pct'] = df['hl_spread'] / df['close']
        
        # Open-Close spread
        if 'open' in df.columns:
            df['oc_spread'] = df['close'] - df['open']
            df['oc_spread_pct'] = df['oc_spread'] / df['open']
        
        return df
    
    def create_volatility_features(self, df: pd.DataFrame, windows: List[int] = [1, 3, 6, 12, 24]) -> pd.DataFrame:
        """
        Create volatility features (target variable and predictors)
        
        Args:
            df: DataFrame with price data
            windows: Rolling window sizes in hours
            
        Returns:
            DataFrame with volatility features
        """
        logger.info("Creating volatility features...")
        df = df.copy()
        
        if 'close' not in df.columns:
            raise ValueError("DataFrame must have 'close' column")
        
        # Calculate returns
        df['returns'] = df['close'].pct_change()
        
        # Realized volatility (standard deviation of returns)
        for window in windows:
            df[f'volatility_{window}h'] = df['returns'].rolling(window=window).std()
            df[f'volatility_{window}h'] = df[f'volatility_{window}h'].fillna(0)
        
        # Target: next hour volatility (shifted back)
        df['target_volatility'] = df['volatility_1h'].shift(-1)
        
        # Additional volatility metrics
        df['abs_returns'] = df['returns'].abs()
        df['squared_returns'] = df['returns'] ** 2
        
        return df
    
    def create_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create volume-based features
        
        Args:
            df: DataFrame with volume data
            
        Returns:
            DataFrame with volume features
        """
        logger.info("Creating volume features...")
        df = df.copy()
        
        if 'volume' not in df.columns:
            logger.warning("No volume column found, skipping volume features")
            return df
        
        # Volume changes
        df['volume_change'] = df['volume'].diff()
        df['volume_change_pct'] = df['volume'].pct_change()
        
        # Rolling volume statistics
        for window in [3, 6, 12, 24]:
            df[f'volume_ma_{window}h'] = df['volume'].rolling(window=window).mean()
            df[f'volume_std_{window}h'] = df['volume'].rolling(window=window).std()
        
        # Volume ratio
        df['volume_ratio'] = df['volume'] / df['volume_ma_24h']
        
        return df
    
    def create_lag_features(self, df: pd.DataFrame, columns: List[str], lags: List[int] = [1, 2, 3, 6]) -> pd.DataFrame:
        """
        Create lag features
        
        Args:
            df: DataFrame
            columns: Columns to create lags for
            lags: Lag periods
            
        Returns:
            DataFrame with lag features
        """
        logger.info("Creating lag features...")
        df = df.copy()
        
        for col in columns:
            if col in df.columns:
                for lag in lags:
                    df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        return df
    
    def create_rolling_features(self, df: pd.DataFrame, columns: List[str], windows: List[int] = [3, 6, 12, 24]) -> pd.DataFrame:
        """
        Create rolling window features
        
        Args:
            df: DataFrame
            columns: Columns to create rolling features for
            windows: Rolling window sizes
            
        Returns:
            DataFrame with rolling features
        """
        logger.info("Creating rolling features...")
        df = df.copy()
        
        for col in columns:
            if col in df.columns:
                for window in windows:
                    df[f'{col}_ma_{window}h'] = df[col].rolling(window=window).mean()
                    df[f'{col}_std_{window}h'] = df[col].rolling(window=window).std()
                    df[f'{col}_min_{window}h'] = df[col].rolling(window=window).min()
                    df[f'{col}_max_{window}h'] = df[col].rolling(window=window).max()
        
        return df
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all transformations
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Transformed DataFrame
        """
        logger.info("Starting data transformation...")
        
        # Clean data
        df = self.clean_data(df)
        
        # Create features
        df = self.create_time_features(df)
        df = self.create_price_features(df)
        df = self.create_volatility_features(df)
        df = self.create_volume_features(df)
        
        # Create lag features for key columns
        key_columns = ['close', 'returns', 'volume']
        df = self.create_lag_features(df, key_columns)
        
        # Create rolling features
        rolling_columns = ['close', 'returns', 'volume']
        df = self.create_rolling_features(df, rolling_columns)
        
        # Drop rows with NaN (from lag and rolling features)
        initial_rows = len(df)
        df = df.dropna()
        logger.info(f"Dropped {initial_rows - len(df)} rows with NaN values")
        
        # Store feature columns (excluding target and metadata)
        exclude_cols = ['target_volatility', 'timestamp', 'symbol', 'interval', 'collection_timestamp']
        self.feature_columns = [col for col in df.columns if col not in exclude_cols]
        
        logger.info(f"Transformation complete: {len(df)} rows, {len(self.feature_columns)} features")
        
        return df


def transform_stock_data(
    input_filepath: str,
    output_filepath: str
) -> str:
    """
    Main transformation function for Airflow
    
    Args:
        input_filepath: Path to raw data file
        output_filepath: Path to save transformed data
        
    Returns:
        Path to saved transformed data file
    """
    logger.info(f"Loading data from {input_filepath}")
    
    # Load data
    if input_filepath.endswith('.parquet'):
        df = pd.read_parquet(input_filepath)
    elif input_filepath.endswith('.csv'):
        df = pd.read_csv(input_filepath)
    else:
        raise ValueError(f"Unsupported file format: {input_filepath}")
    
    # Transform
    transformer = StockDataTransformer()
    df_transformed = transformer.transform(df)
    
    # Save
    Path(output_filepath).parent.mkdir(parents=True, exist_ok=True)
    df_transformed.to_parquet(output_filepath, index=False)
    
    logger.info(f"Transformed data saved to {output_filepath}")
    
    return output_filepath


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2:
        input_path = sys.argv[1]
        output_path = sys.argv[2]
        transform_stock_data(input_path, output_path)
        print(f"Transformation complete: {output_path}")
    else:
        print("Usage: python transform.py <input_file> <output_file>")

