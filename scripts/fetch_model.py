"""
Script to fetch the best model from MLflow registry
"""
import os
import sys
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import joblib
from pathlib import Path

def fetch_model():
    """Fetch best model from MLflow and save locally"""
    mlflow_tracking_uri = os.getenv('MLFLOW_TRACKING_URI')
    if not mlflow_tracking_uri:
        print("MLFLOW_TRACKING_URI not set")
        sys.exit(1)
    
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    client = MlflowClient()
    
    # Get production model
    try:
        prod_models = client.get_latest_versions(
            "StockVolatilityPredictor",
            stages=["Production"]
        )
        
        if not prod_models:
            # Fallback to latest version
            prod_models = client.get_latest_versions("StockVolatilityPredictor")
        
        if prod_models:
            model_uri = f"models:/StockVolatilityPredictor/{prod_models[0].version}"
            print(f"Fetching model: {model_uri}")
            
            # Load model
            model = mlflow.sklearn.load_model(model_uri)
            
            # Save locally
            model_dir = Path("models")
            model_dir.mkdir(exist_ok=True)
            
            model_path = model_dir / "model.joblib"
            joblib.dump(model, model_path)
            print(f"Model saved to {model_path}")
            
            # Try to fetch feature columns
            try:
                run_id = prod_models[0].run_id
                artifacts_path = mlflow.artifacts.download_artifacts(
                    run_id=run_id,
                    artifact_path="artifacts/features.pkl"
                )
                
                import shutil
                features_path = model_dir / "features.pkl"
                shutil.copy(artifacts_path, features_path)
                print(f"Features saved to {features_path}")
            except Exception as e:
                print(f"Could not fetch features: {e}")
        else:
            print("No model found in registry")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error fetching model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    fetch_model()

