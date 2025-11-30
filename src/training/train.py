"""
Model training script with MLflow tracking
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
import logging
import joblib
from pathlib import Path
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config():
    """Load configuration from config file"""
    config_path = os.path.join(
        os.path.dirname(__file__), '../../config/config.yaml'
    )
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def prepare_data(df: pd.DataFrame, config: dict) -> tuple:
    """
    Prepare data for training
    
    Args:
        df: DataFrame with features
        config: Configuration dictionary
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    logger.info("Preparing data for training...")
    
    # Identify target and features
    target_col = config['model']['target']
    
    # If target is 'volatility', use the target_volatility column
    if target_col == 'volatility':
        target_col = 'target_volatility'
    
    # Get feature columns (exclude metadata and target)
    exclude_cols = [
        'target_volatility', 'timestamp', 'symbol', 
        'interval', 'collection_timestamp', 'returns'
    ]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Remove any remaining NaN
    df_clean = df.dropna()
    
    X = df_clean[feature_cols]
    y = df_clean[target_col]
    
    # Split data
    test_size = config['model']['test_size']
    random_state = config['model']['random_state']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=False
    )
    
    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    logger.info(f"Features: {len(feature_cols)}")
    
    return X_train, X_test, y_train, y_test, feature_cols


def train_model(X_train, y_train, config: dict):
    """
    Train the model
    
    Args:
        X_train: Training features
        y_train: Training target
        config: Configuration dictionary
        
    Returns:
        Trained model
    """
    logger.info("Training model...")
    
    # Get hyperparameters
    hyperparams = config['model']['hyperparameters']
    
    # Use Gradient Boosting for time series
    model = GradientBoostingRegressor(
        n_estimators=hyperparams.get('n_estimators', 100),
        max_depth=hyperparams.get('max_depth', 10),
        learning_rate=hyperparams.get('learning_rate', 0.1),
        random_state=config['model']['random_state']
    )
    
    model.fit(X_train, y_train)
    
    logger.info("Model training completed")
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        
    Returns:
        Dictionary of metrics
    """
    logger.info("Evaluating model...")
    
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Additional metrics
    mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100
    
    metrics = {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape
    }
    
    logger.info(f"RMSE: {rmse:.6f}")
    logger.info(f"MAE: {mae:.6f}")
    logger.info(f"R²: {r2:.6f}")
    logger.info(f"MAPE: {mape:.2f}%")
    
    return metrics


def main(data_path: str):
    """
    Main training function
    
    Args:
        data_path: Path to processed data file
    """
    # Load configuration
    config = load_config()
    
    # Set up MLflow
    mlflow_tracking_uri = os.getenv('MLFLOW_TRACKING_URI')
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        logger.info(f"MLflow tracking URI: {mlflow_tracking_uri}")
    else:
        logger.warning("MLFLOW_TRACKING_URI not set, using local tracking")
    
    experiment_name = config['mlflow']['experiment_name']
    mlflow.set_experiment(experiment_name)
    
    # Load data
    logger.info(f"Loading data from {data_path}")
    df = pd.read_parquet(data_path)
    logger.info(f"Loaded {len(df)} rows")
    
    # Prepare data
    X_train, X_test, y_train, y_test, feature_cols = prepare_data(df, config)
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"training_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"):
        # Log parameters
        mlflow.log_param("data_path", data_path)
        mlflow.log_param("n_samples", len(df))
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_param("test_size", config['model']['test_size'])
        
        # Log hyperparameters
        hyperparams = config['model']['hyperparameters']
        for key, value in hyperparams.items():
            mlflow.log_param(key, value)
        
        # Train model
        model = train_model(X_train, y_train, config)
        
        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test)
        
        # Log metrics
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
        
        # Log model
        mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name="StockVolatilityPredictor"
        )
        
        # Save feature columns for inference
        feature_path = "features.pkl"
        joblib.dump(feature_cols, feature_path)
        mlflow.log_artifact(feature_path, "artifacts")
        
        # Save model locally as well
        model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)
        model_path = model_dir / "model.joblib"
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")
        
        logger.info("Training completed successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train stock volatility prediction model")
    parser.add_argument(
        '--data-path',
        type=str,
        required=True,
        help='Path to processed data file'
    )
    
    args = parser.parse_args()
    
    try:
        main(args.data_path)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)

