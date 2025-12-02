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
    
    # Handle Dagshub authentication
    dagshub_username = os.getenv('DAGSHUB_USERNAME')
    dagshub_token = os.getenv('DAGSHUB_TOKEN')
    
    if dagshub_username and dagshub_token and 'dagshub.com' in mlflow_tracking_uri:
        from urllib.parse import urlparse, urlunparse
        parsed = urlparse(mlflow_tracking_uri)
        auth_uri = urlunparse((
            parsed.scheme,
            f"{dagshub_username}:{dagshub_token}@{parsed.netloc}",
            parsed.path,
            parsed.params,
            parsed.query,
            parsed.fragment
        ))
        mlflow.set_tracking_uri(auth_uri)
    else:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    client = MlflowClient()
    
    # Try to get model from registry first, fallback to latest experiment run
    model_uri = None
    run_id = None
    
    try:
        # Try Model Registry (may not work with Dagshub)
        prod_models = client.get_latest_versions(
            "StockVolatilityPredictor",
            stages=["Production"]
        )
        if prod_models:
            model_uri = f"models:/StockVolatilityPredictor/Production"
            run_id = prod_models[0].run_id
            print(f"Found model in registry: {model_uri}")
    except Exception:
        # Fallback: Get latest run from experiment (works with Dagshub)
        try:
            experiment = mlflow.get_experiment_by_name("stock_volatility_prediction")
            if experiment:
                runs = client.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    order_by=["start_time DESC"],
                    max_results=1
                )
                if runs:
                    run_id = runs[0].info.run_id
                    model_uri = f"runs:/{run_id}/model"
                    print(f"Found model in latest run: {run_id}")
        except Exception as e:
            print(f"Error fetching model: {e}")
            sys.exit(1)
    
    if not model_uri:
        print("No model found")
        sys.exit(1)
    
    # Load model
    print(f"Fetching model: {model_uri}")
    model = mlflow.sklearn.load_model(model_uri)
    
    # Save locally
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    
    model_path = model_dir / "model.joblib"
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    # Try to fetch feature columns
    if run_id:
        try:
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

if __name__ == "__main__":
    fetch_model()
