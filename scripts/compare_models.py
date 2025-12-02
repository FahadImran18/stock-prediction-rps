"""
Script to compare new model with production model using CML
"""
import os
import sys
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pandas as pd
import time
from urllib.parse import urlparse, urlunparse

def check_dagshub_connectivity(mlflow_tracking_uri, max_retries=3, retry_delay=5):
    """Check if Dagshub/MLflow is accessible"""
    if 'dagshub.com' not in mlflow_tracking_uri:
        return True  # Not Dagshub, assume accessible
    
    import requests
    parsed = urlparse(mlflow_tracking_uri)
    base_url = f"{parsed.scheme}://{parsed.netloc}"
    
    for attempt in range(max_retries):
        try:
            response = requests.get(base_url, timeout=10)
            if response.status_code in [200, 401, 403]:  # 401/403 means server is up
                return True
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                print(f"⚠️ Dagshub connectivity check failed (attempt {attempt + 1}/{max_retries}): {e}")
                print(f"   Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print(f"❌ Dagshub appears to be down or unreachable: {e}")
                return False
    return False

def compare_models():
    """Compare new model with production model"""
    mlflow_tracking_uri = os.getenv('MLFLOW_TRACKING_URI')
    if not mlflow_tracking_uri:
        print("❌ MLFLOW_TRACKING_URI not set")
        sys.exit(1)
    
    print(f"📊 MLflow Tracking URI: {mlflow_tracking_uri}")
    
    # Check Dagshub connectivity if using Dagshub
    if 'dagshub.com' in mlflow_tracking_uri:
        print("🔍 Checking Dagshub connectivity...")
        if not check_dagshub_connectivity(mlflow_tracking_uri):
            print("\n⚠️ ==========================================")
            print("⚠️ WARNING: Dagshub appears to be down")
            print("⚠️ Cannot fetch model metrics for comparison")
            print("⚠️ ==========================================")
            print("\nOptions:")
            print("1. Wait for Dagshub to come back online and re-run")
            print("2. If this is urgent, you may need to manually verify model performance")
            print("\n❌ **REJECT**: Cannot verify model performance - blocking merge")
            print("   This ensures we don't deploy untested models")
            sys.exit(1)
        print("✅ Dagshub is accessible")
    
    # Handle Dagshub authentication
    dagshub_username = os.getenv('DAGSHUB_USERNAME')
    dagshub_token = os.getenv('DAGSHUB_TOKEN')
    
    if dagshub_username and dagshub_token and 'dagshub.com' in mlflow_tracking_uri:
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
        print(f"🔐 Authenticated with Dagshub as: {dagshub_username}")
    else:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        if 'dagshub.com' in mlflow_tracking_uri:
            print("⚠️ DAGSHUB_USERNAME and DAGSHUB_TOKEN not set - authentication may fail")
    
    try:
        client = MlflowClient()
        print("✅ MLflow client initialized")
    except Exception as e:
        print(f"❌ Failed to initialize MLflow client: {e}")
        sys.exit(1)
    
    # Get production model from experiment runs (Dagshub doesn't support Model Registry well)
    # We'll filter for training runs only (not data profile runs)
    prod_metrics = None
    
    # Get training runs from experiment (filter out data profile runs)
    try:
        print("🔍 Fetching experiment: stock_volatility_prediction")
        experiment = mlflow.get_experiment_by_name("stock_volatility_prediction")
        if not experiment:
            print("⚠️ Experiment 'stock_volatility_prediction' not found")
        else:
            print(f"✅ Found experiment: {experiment.experiment_id}")
            print("🔍 Searching for training runs (filtering out data profile runs)...")
            
            # Get all runs and filter for training runs only
            all_runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["start_time DESC"],
                max_results=10  # Get more runs to find training runs
            )
            print(f"📊 Found {len(all_runs)} total run(s) in experiment")
            
            # Filter for training runs (run_name starts with "training_")
            training_runs = [run for run in all_runs if run.info.run_name and run.info.run_name.startswith("training_")]
            print(f"📊 Found {len(training_runs)} training run(s)")
            
            if len(training_runs) >= 2:
                # Use second-to-last training run as "production" baseline
                prod_run = training_runs[1]
                print(f"📌 Using training run {prod_run.info.run_id} as production baseline")
                print(f"   Run name: {prod_run.info.run_name}")
                print(f"   Start time: {prod_run.info.start_time}")
                prod_metrics = {
                    'rmse': prod_run.data.metrics.get('rmse'),
                    'mae': prod_run.data.metrics.get('mae'),
                    'r2': prod_run.data.metrics.get('r2')
                }
                # Validate metrics exist and are reasonable
                if not all([prod_metrics['rmse'] is not None, prod_metrics['mae'] is not None, prod_metrics['r2'] is not None]):
                    print("⚠️ Production run exists but metrics are missing")
                    prod_metrics = None
                elif prod_metrics['rmse'] == 0 and prod_metrics['mae'] == 0 and prod_metrics['r2'] == 1.0:
                    print("⚠️ Production metrics look suspicious (perfect scores) - may indicate data/evaluation issue")
                    # Still use it, but warn
                else:
                    print(f"✅ Production metrics: RMSE={prod_metrics['rmse']:.6f}, MAE={prod_metrics['mae']:.6f}, R²={prod_metrics['r2']:.6f}")
            elif len(training_runs) == 1:
                print("ℹ️ Only 1 training run found - this will be the baseline for future comparisons")
            else:
                print("⚠️ No training runs found in experiment")
                print("   Found runs:")
                for run in all_runs[:5]:  # Show first 5 runs
                    print(f"     - {run.info.run_name} (ID: {run.info.run_id})")
    except Exception as e:
        print(f"❌ Could not fetch production model: {e}")
        import traceback
        print(traceback.format_exc())
    
    # Get latest training run (most recent training run, not data profile)
    latest_metrics = None
    try:
        print("🔍 Fetching latest training run...")
        experiment = mlflow.get_experiment_by_name("stock_volatility_prediction")
        if experiment:
            # Get all runs and filter for training runs
            all_runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["start_time DESC"],
                max_results=10
            )
            # Filter for training runs only
            training_runs = [run for run in all_runs if run.info.run_name and run.info.run_name.startswith("training_")]
            
            if training_runs:
                latest_run = training_runs[0]  # Most recent training run
                print(f"📌 Latest training run: {latest_run.info.run_id}")
                print(f"   Run name: {latest_run.info.run_name}")
                print(f"   Start time: {latest_run.info.start_time}")
                latest_metrics = {
                    'rmse': latest_run.data.metrics.get('rmse'),
                    'mae': latest_run.data.metrics.get('mae'),
                    'r2': latest_run.data.metrics.get('r2')
                }
                # Validate metrics exist
                if not all([latest_metrics['rmse'] is not None, latest_metrics['mae'] is not None, latest_metrics['r2'] is not None]):
                    latest_metrics = None
                    print("❌ Latest training run exists but metrics are missing")
                    print(f"   Available metrics: {list(latest_run.data.metrics.keys())}")
                elif latest_metrics['rmse'] == 0 and latest_metrics['mae'] == 0 and latest_metrics['r2'] == 1.0:
                    print("⚠️ Latest metrics look suspicious (perfect scores):")
                    print(f"   RMSE={latest_metrics['rmse']:.6f}, MAE={latest_metrics['mae']:.6f}, R²={latest_metrics['r2']:.6f}")
                    print("   This may indicate:")
                    print("   - Data leakage in features")
                    print("   - Test set too small or identical to training set")
                    print("   - Evaluation issue")
                    print("   ⚠️ Proceeding with comparison, but please investigate")
                else:
                    print(f"✅ Latest metrics: RMSE={latest_metrics['rmse']:.6f}, MAE={latest_metrics['mae']:.6f}, R²={latest_metrics['r2']:.6f}")
            else:
                print("⚠️ No training runs found in experiment")
                print("   Available runs:")
                for run in all_runs[:5]:
                    print(f"     - {run.info.run_name} (ID: {run.info.run_id})")
        else:
            print("⚠️ Experiment 'stock_volatility_prediction' not found")
    except Exception as e:
        print(f"❌ Could not fetch latest model: {e}")
        import traceback
        print(traceback.format_exc())
    
    # Generate comparison report
    print("\n# Model Comparison Report\n")
    
    # Check if we're comparing the same run (no new runs added)
    if prod_metrics and latest_metrics:
        # Get run IDs to check if they're the same
        try:
            experiment = mlflow.get_experiment_by_name("stock_volatility_prediction")
            if experiment:
                all_runs = client.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    order_by=["start_time DESC"],
                    max_results=10
                )
                training_runs = [run for run in all_runs if run.info.run_name and run.info.run_name.startswith("training_")]
                
                if len(training_runs) == 1:
                    print("⚠️ Only 1 training run found in experiment")
                    print("   This means no new model has been trained since the last comparison")
                    print("   The training pipeline may not have run, or Dagshub may not have received the new run")
                    print("\n❌ **REJECT**: No new model to compare")
                    print("   Please ensure:")
                    print("   1. The Airflow DAG completed successfully")
                    print("   2. The training script logged metrics to MLflow/Dagshub")
                    print("   3. Dagshub is accessible and receiving new runs")
                    sys.exit(1)
                elif len(training_runs) >= 2:
                    latest_run_id = training_runs[0].info.run_id
                    prod_run_id = training_runs[1].info.run_id
                    if latest_run_id == prod_run_id:
                        print("⚠️ Latest training run and production run are the same")
                        print("   No new model has been trained")
                        print("\n❌ **REJECT**: No new model to compare")
                        sys.exit(1)
        except Exception as e:
            print(f"⚠️ Could not verify run uniqueness: {e}")
            # Continue with comparison anyway
    
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

