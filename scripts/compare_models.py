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
    
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    client = MlflowClient()
    
    # Get production model
    try:
        prod_models = client.get_latest_versions(
            "StockVolatilityPredictor",
            stages=["Production"]
        )
        if prod_models:
            prod_run = client.get_run(prod_models[0].run_id)
            prod_metrics = {
                'rmse': prod_run.data.metrics.get('rmse', 0),
                'mae': prod_run.data.metrics.get('mae', 0),
                'r2': prod_run.data.metrics.get('r2', 0)
            }
        else:
            prod_metrics = None
    except Exception as e:
        print(f"Could not fetch production model: {e}")
        prod_metrics = None
    
    # Get latest model (from current run)
    try:
        latest_models = client.get_latest_versions("StockVolatilityPredictor")
        if latest_models:
            latest_run = client.get_run(latest_models[0].run_id)
            latest_metrics = {
                'rmse': latest_run.data.metrics.get('rmse', 0),
                'mae': latest_run.data.metrics.get('mae', 0),
                'r2': latest_run.data.metrics.get('r2', 0)
            }
        else:
            latest_metrics = None
    except Exception as e:
        print(f"Could not fetch latest model: {e}")
        latest_metrics = None
    
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
        improvements = sum([
            latest_metrics['r2'] > prod_metrics['r2'],
            latest_metrics['rmse'] < prod_metrics['rmse'],
            latest_metrics['mae'] < prod_metrics['mae']
        ])
        
        if improvements >= 2:
            print("✅ **APPROVE**: New model performs better on 2+ metrics")
        else:
            print("❌ **REJECT**: New model does not improve performance")
            sys.exit(1)
    else:
        print("⚠️ Could not compare models - missing metrics")
        if not prod_metrics:
            print("- Production model metrics not available")
        if not latest_metrics:
            print("- Latest model metrics not available")

if __name__ == "__main__":
    compare_models()

