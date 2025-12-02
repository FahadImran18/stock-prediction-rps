"""
Data extraction module for fetching stock data from stockdata.org API
"""
import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional
import time
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockDataExtractor:
    """Extract stock data from stockdata.org API"""
    
    def __init__(self, api_key: str, base_url: str = "https://api.stockdata.org/v1"):
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
        # Note: stockdata.org uses api_token as GET parameter, not Bearer token
    
    def fetch_stock_data(
        self, 
        symbol: str, 
        interval: str = "hour",
        lookback_days: int = 30,
        use_adjusted: bool = False
    ) -> pd.DataFrame:
        """
        Fetch historical stock data from stockdata.org API
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            interval: Data interval ('hour' or 'minute' for intraday, 'day' for EOD)
            lookback_days: Number of days to look back (max 180 for hour, 7 for minute)
            use_adjusted: Use adjusted intraday data (requires Standard plan or above)
            
        Returns:
            DataFrame with stock data
        """
        logger.info(f"Fetching data for {symbol} with {interval} interval")
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=min(lookback_days, 180))  # Max 180 days for hour interval
        
        # Format dates for API (Y-m-d format)
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")
        
        # Select endpoint based on interval
        if interval in ['hour', 'minute']:
            # Use intraday endpoint
            if use_adjusted:
                endpoint = "/data/intraday/adjusted"
            else:
                endpoint = "/data/intraday"
        else:
            # Use end-of-day endpoint
            endpoint = "/data/eod"
        
        url = f"{self.base_url}{endpoint}"
        
        # Build parameters according to stockdata.org API documentation
        params = {
            "api_token": self.api_key,
            "symbols": symbol,
            "interval": interval,
            "date_from": start_str,
            "date_to": end_str,
            "sort": "asc"  # Sort ascending for chronological order
        }
        
        try:
            logger.info(f"Requesting data from {url}")
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Handle API errors
            if 'error' in data:
                error_code = data['error'].get('code', 'unknown')
                error_msg = data['error'].get('message', 'Unknown error')
                raise ValueError(f"API Error ({error_code}): {error_msg}")
            
            # Convert to DataFrame based on actual API response structure
            if 'data' not in data or not data['data']:
                logger.warning("No data returned from API")
                return pd.DataFrame()
            
            # Parse the response structure
            # Intraday format: {"data": [{"date": "...", "ticker": "...", "data": {"open": ..., ...}}]}
            # EOD format: {"data": [{"date": "...", "open": ..., "high": ..., ...}]}
            records = []
            for item in data['data']:
                # Check if it's intraday format (nested data) or EOD format (flat)
                if 'data' in item and isinstance(item['data'], dict):
                    # Intraday format
                    record = {
                        'timestamp': pd.to_datetime(item['date']),
                        'symbol': item.get('ticker', symbol),
                        'open': item['data']['open'],
                        'high': item['data']['high'],
                        'low': item['data']['low'],
                        'close': item['data']['close'],
                        'volume': item['data']['volume']
                    }
                    
                    # Add extended hours flag if available
                    if 'is_extended_hours' in item['data']:
                        record['is_extended_hours'] = item['data']['is_extended_hours']
                else:
                    # EOD format (flat structure)
                    record = {
                        'timestamp': pd.to_datetime(item['date']),
                        'symbol': item.get('ticker', symbol),
                        'open': item['open'],
                        'high': item['high'],
                        'low': item['low'],
                        'close': item['close'],
                        'volume': item['volume']
                    }
                
                records.append(record)
            
            df = pd.DataFrame(records)
            
            # Add metadata
            df['collection_timestamp'] = datetime.now().isoformat()
            df['interval'] = interval
            
            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            logger.info(f"Successfully fetched {len(df)} records")
            logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            
            return df
            
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP Error {e.response.status_code}: {e.response.text}")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching data: {e}")
            raise
        except KeyError as e:
            logger.error(f"Unexpected response structure. Missing key: {e}")
            logger.error(f"Response: {data}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise
    
    def save_raw_data(self, df: pd.DataFrame, output_path: str) -> str:
        """
        Save raw data with timestamp
        
        Args:
            df: DataFrame to save
            output_path: Directory to save data
            
        Returns:
            Path to saved file
        """
        Path(output_path).mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"raw_stock_data_{timestamp}.parquet"
        filepath = os.path.join(output_path, filename)
        
        df.to_parquet(filepath, index=False)
        logger.info(f"Raw data saved to {filepath}")
        
        return filepath


def extract_stock_data(
    api_key: str,
    symbol: str = "AAPL",
    interval: str = "hour",
    lookback_days: int = 30,
    output_path: str = "data/raw",
    use_adjusted: bool = False
) -> str:
    """
    Main extraction function for Airflow
    
    Args:
        api_key: Stockdata.org API key
        symbol: Stock symbol (e.g., 'AAPL')
        interval: Data interval ('hour', 'minute', or 'day')
        lookback_days: Lookback period (max 180 for hour, 7 for minute)
        output_path: Output directory
        use_adjusted: Use adjusted intraday data (requires Standard plan or above)
        
    Returns:
        Path to saved raw data file
    """
    extractor = StockDataExtractor(api_key)
    df = extractor.fetch_stock_data(symbol, interval, lookback_days, use_adjusted)
    
    if df.empty:
        raise ValueError(f"No data returned for symbol {symbol}")
    
    filepath = extractor.save_raw_data(df, output_path)
    
    return filepath


if __name__ == "__main__":
    # Test extraction
    api_key = os.getenv("STOCKDATA_API_KEY")
    if not api_key:
        raise ValueError("STOCKDATA_API_KEY environment variable not set")
    
    filepath = extract_stock_data(api_key)
    print(f"Data extracted and saved to: {filepath}")

# Test CI Pipeline