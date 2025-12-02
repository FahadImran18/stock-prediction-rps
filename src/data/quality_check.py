"""
Data quality checks and validation
"""
import pandas as pd
import logging
from typing import Dict, Tuple, List
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataQualityChecker:
    """Perform data quality checks on extracted data"""
    
    def __init__(
        self,
        max_null_percentage: float = 1.0,
        required_columns: List[str] = None,
        min_rows: int = 100
    ):
        self.max_null_percentage = max_null_percentage
        self.required_columns = required_columns or []
        self.min_rows = min_rows
        self.checks_passed = True
        self.errors = []
    
    def check_null_percentage(self, df: pd.DataFrame) -> bool:
        """
        Check if null percentage exceeds threshold
        
        Args:
            df: DataFrame to check
            
        Returns:
            True if check passes
        """
        total_cells = df.size
        null_cells = df.isnull().sum().sum()
        null_percentage = (null_cells / total_cells) * 100
        
        logger.info(f"Null percentage: {null_percentage:.2f}%")
        
        if null_percentage > self.max_null_percentage:
            error_msg = (
                f"Data quality check FAILED: "
                f"Null percentage ({null_percentage:.2f}%) exceeds "
                f"threshold ({self.max_null_percentage}%)"
            )
            logger.error(error_msg)
            self.errors.append(error_msg)
            self.checks_passed = False
            return False
        
        logger.info("Null percentage check PASSED")
        return True
    
    def check_required_columns(self, df: pd.DataFrame) -> bool:
        """
        Check if all required columns are present
        
        Args:
            df: DataFrame to check
            
        Returns:
            True if check passes
        """
        missing_columns = set(self.required_columns) - set(df.columns)
        
        if missing_columns:
            error_msg = (
                f"Data quality check FAILED: "
                f"Missing required columns: {missing_columns}"
            )
            logger.error(error_msg)
            self.errors.append(error_msg)
            self.checks_passed = False
            return False
        
        logger.info("Required columns check PASSED")
        return True
    
    def check_min_rows(self, df: pd.DataFrame) -> bool:
        """
        Check if DataFrame has minimum required rows
        
        Args:
            df: DataFrame to check
            
        Returns:
            True if check passes
        """
        if len(df) < self.min_rows:
            error_msg = (
                f"Data quality check FAILED: "
                f"Number of rows ({len(df)}) is less than "
                f"minimum required ({self.min_rows})"
            )
            logger.error(error_msg)
            self.errors.append(error_msg)
            self.checks_passed = False
            return False
        
        logger.info(f"Minimum rows check PASSED ({len(df)} rows)")
        return True
    
    def check_schema(self, df: pd.DataFrame) -> bool:
        """
        Basic schema validation
        
        Args:
            df: DataFrame to check
            
        Returns:
            True if check passes
        """
        # Check for numeric columns that should be numeric
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        present_numeric = [col for col in numeric_columns if col in df.columns]
        
        for col in present_numeric:
            if not pd.api.types.is_numeric_dtype(df[col]):
                error_msg = (
                    f"Data quality check FAILED: "
                    f"Column '{col}' should be numeric but is {df[col].dtype}"
                )
                logger.error(error_msg)
                self.errors.append(error_msg)
                self.checks_passed = False
                return False
        
        logger.info("Schema validation PASSED")
        return True
    
    def validate(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Run all quality checks
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Tuple of (checks_passed, list_of_errors)
        """
        logger.info("Starting data quality checks...")
        
        self.checks_passed = True
        self.errors = []
        
        # Run all checks
        self.check_null_percentage(df)
        self.check_required_columns(df)
        self.check_min_rows(df)
        self.check_schema(df)
        
        if self.checks_passed:
            logger.info("All data quality checks PASSED")
        else:
            logger.error("Data quality checks FAILED")
            for error in self.errors:
                logger.error(f"  - {error}")
        
        return self.checks_passed, self.errors


def validate_data_quality(
    filepath: str,
    max_null_percentage: float = 1.0,
    required_columns: List[str] = None,
    min_rows: int = 100
) -> bool:
    """
    Main validation function for Airflow
    
    Args:
        filepath: Path to data file
        max_null_percentage: Maximum allowed null percentage
        required_columns: List of required columns
        min_rows: Minimum number of rows
        
    Returns:
        True if validation passes, raises exception otherwise
    """
    if required_columns is None:
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    
    # Load data
    if filepath.endswith('.parquet'):
        df = pd.read_parquet(filepath)
    elif filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
    else:
        raise ValueError(f"Unsupported file format: {filepath}")
    
    # Run validation
    checker = DataQualityChecker(
        max_null_percentage=max_null_percentage,
        required_columns=required_columns,
        min_rows=min_rows
    )
    
    checks_passed, errors = checker.validate(df)
    
    if not checks_passed:
        error_summary = "\n".join(errors)
        raise ValueError(
            f"Data quality check failed:\n{error_summary}\n"
            f"DAG execution stopped."
        )
    
    return True


if __name__ == "__main__":
    # Test quality check
    import sys
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        validate_data_quality(filepath)
        print("Quality checks passed!")
    else:
        print("Usage: python quality_check.py <data_file_path>")

