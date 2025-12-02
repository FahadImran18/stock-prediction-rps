"""
Script to compare new model with production model using CML
"""
import os
import sys
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pandas as pd

def compare_models():
    """Compare new model with production model"""
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
    
    # Get production model (try registry first, fallback to latest experiment run)
    prod_metrics = None
    try:
        # Try Model Registry (may not work with Dagshub)
        prod_models = client.get_latest_versions(
            "StockVolatilityPredictor",
            stages=["Production"]
        )
        if prod_models:
            prod_run = client.get_run(prod_models[0].run_id)
            prod_metrics = {
                'rmse': prod_run.data.metrics.get('rmse'),
                'mae': prod_run.data.metrics.get('mae'),
                'r2': prod_run.data.metrics.get('r2')
            }
            # Validate metrics exist
            if not all([prod_metrics['rmse'] is not None, prod_metrics['mae'] is not None, prod_metrics['r2'] is not None]):
                prod_metrics = None
    except Exception:
        pass  # Fall through to experiment-based lookup
    
    # Fallback: Get latest run from experiment (works with Dagshub)
    if not prod_metrics:
        try:
            experiment = mlflow.get_experiment_by_name("stock_volatility_prediction")
            if experiment:
                runs = client.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    order_by=["start_time DESC"],
                    max_results=2  # Get last 2 runs
                )
                if len(runs) >= 2:
                    # Use second-to-last as "production" baseline
                    prod_run = runs[1]
                    prod_metrics = {
                        'rmse': prod_run.data.metrics.get('rmse'),
                        'mae': prod_run.data.metrics.get('mae'),
                        'r2': prod_run.data.metrics.get('r2')
                    }
                    # Validate metrics exist
                    if not all([prod_metrics['rmse'] is not None, prod_metrics['mae'] is not None, prod_metrics['r2'] is not None]):
                        prod_metrics = None
        except Exception as e:
            print(f"⚠️ Could not fetch production model: {e}")
    
    # Get latest model (most recent run)
    latest_metrics = None
    try:
        experiment = mlflow.get_experiment_by_name("stock_volatility_prediction")
        if experiment:
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["start_time DESC"],
                max_results=1
            )
            if runs:
                latest_run = runs[0]
                latest_metrics = {
                    'rmse': latest_run.data.metrics.get('rmse'),
                    'mae': latest_run.data.metrics.get('mae'),
                    'r2': latest_run.data.metrics.get('r2')
                }
                # Validate metrics exist
                if not all([latest_metrics['rmse'] is not None, latest_metrics['mae'] is not None, latest_metrics['r2'] is not None]):
                    latest_metrics = None
                    print("⚠️ Latest model run exists but metrics are missing")
    except Exception as e:
        print(f"⚠️ Could not fetch latest model: {e}")
    
    # Generate comparison report
    print("# Model Comparison Report\n")
    
    if prod_metrics and latest_metrics:
        print("## Metrics Comparison\n")
        print("| Metric | Production | Latest | Change | Status |")
        print("|--------|------------|--------|--------|--------|")
        
        for metric in ['rmse', 'mae', 'r2']:
            prod_val = prod_metrics[metric]
            latest_val = latest_metrics[metric]
            
            if metric == 'r2':
                # Higher is better for R2
                change = latest_val - prod_val
                status = "✅ Better" if change > 0 else "❌ Worse"
            else:
                # Lower is better for RMSE and MAE
                change = prod_val - latest_val
                status = "✅ Better" if change > 0 else "❌ Worse"
            
            print(f"| {metric.upper()} | {prod_val:.6f} | {latest_val:.6f} | {change:+.6f} | {status} |")
        
        # Overall recommendation
        print("\n## Recommendation\n")
        
        # Count improvements (model must be better on at least 2 metrics)
        improvements = sum([
            latest_metrics['r2'] > prod_metrics['r2'],  # Higher R2 is better
            latest_metrics['rmse'] < prod_metrics['rmse'],  # Lower RMSE is better
            latest_metrics['mae'] < prod_metrics['mae']  # Lower MAE is better
        ])
        
        # Calculate percentage changes
        r2_change_pct = ((latest_metrics['r2'] - prod_metrics['r2']) / abs(prod_metrics['r2']) * 100) if prod_metrics['r2'] != 0 else 0
        rmse_change_pct = ((prod_metrics['rmse'] - latest_metrics['rmse']) / prod_metrics['rmse'] * 100) if prod_metrics['rmse'] != 0 else 0
        mae_change_pct = ((prod_metrics['mae'] - latest_metrics['mae']) / prod_metrics['mae'] * 100) if prod_metrics['mae'] != 0 else 0
        
        print(f"**Improvements**: {improvements}/3 metrics improved")
        print(f"- R² change: {r2_change_pct:+.2f}%")
        print(f"- RMSE change: {rmse_change_pct:+.2f}%")
        print(f"- MAE change: {mae_change_pct:+.2f}%")
        print()
        
        # Decision: Model must improve on at least 2 metrics (per project requirements)
        if improvements >= 2:
            print("✅ **APPROVE**: New model performs better on 2+ metrics")
            print("✅ **MERGE APPROVED**: Model meets performance requirements")
            sys.exit(0)  # Success - allow merge
        else:
            print("❌ **REJECT**: New model does not improve performance")
            print(f"❌ Only {improvements}/3 metrics improved (required: 2+)")
            print("❌ **MERGE BLOCKED**: Model performance must improve before merging")
            sys.exit(1)  # Fail - block merge
    else:
        print("⚠️ Could not compare models - missing metrics")
        if not prod_metrics:
            print("- Production model metrics not available")
            print("  This might be the first model run. Creating baseline...")
            # For first run, allow it to pass (it becomes the baseline)
            if latest_metrics:
                print("✅ **APPROVE**: First model run - setting as baseline")
                print("✅ **MERGE APPROVED**: This model will serve as the production baseline")
                sys.exit(0)  # Allow first model to pass
        if not latest_metrics:
            print("- Latest model metrics not available")
            print("  The training run may have failed or metrics were not logged")
        
        # If we can't compare and it's not the first run, fail
        print("\n❌ **REJECT**: Cannot verify model performance - blocking merge")
        print("❌ Please ensure:")
        print("   1. Training completed successfully")
        print("   2. Metrics (rmse, mae, r2) were logged to MLflow")
        print("   3. MLflow tracking URI and credentials are correct")
        sys.exit(1)

if __name__ == "__main__":
    compare_models()

