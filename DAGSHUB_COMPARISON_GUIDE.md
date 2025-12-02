# Dagshub Model Comparison Guide

## ✅ Yes, It Checks Dagshub Experiments!

The `compare_models.py` script **directly queries Dagshub's MLflow experiments** to compare models.

## 🔍 How It Works

### 1. **Connection to Dagshub**

The script:
1. Uses `MLFLOW_TRACKING_URI` (e.g., `https://dagshub.com/username/repo.mlflow`)
2. Authenticates with `DAGSHUB_USERNAME` and `DAGSHUB_TOKEN`
3. Connects to Dagshub's MLflow API

### 2. **Experiment Lookup**

```python
experiment = mlflow.get_experiment_by_name("stock_volatility_prediction")
```

- Looks for experiment named `"stock_volatility_prediction"`
- This is the **same experiment** that `train.py` logs to
- All training runs are stored in this experiment

### 3. **Run Comparison**

The script fetches runs from the experiment:
- **Production model**: 2nd most recent run (baseline)
- **Latest model**: Most recent run (newly trained)

```python
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["start_time DESC"],
    max_results=2  # Get last 2 runs
)
```

## ⚠️ Common Issues

### Issue 1: "No new experiment added after first run"

**Symptoms**:
- Only 1 run exists in experiment
- Script says "Only 1 run found"

**Causes**:
1. **Training didn't run**: Airflow DAG may have failed
2. **Training didn't log to MLflow**: Check training logs
3. **Dagshub didn't receive the run**: Network/authentication issue

**Solutions**:
1. Check Airflow logs for `train_model` task
2. Verify `MLFLOW_TRACKING_URI` is set correctly
3. Verify `DAGSHUB_USERNAME` and `DAGSHUB_TOKEN` are correct
4. Check Dagshub UI to see if runs are appearing

### Issue 2: "Dagshub may be down"

**Symptoms**:
- Connection timeout errors
- "Dagshub appears to be down or unreachable"

**What Happens**:
- ✅ Script now **checks connectivity first** (with retries)
- ✅ Provides clear error messages
- ❌ **Blocks merge** if Dagshub is down (safety measure)

**Why Block Merge?**:
- We can't verify model performance without metrics
- Prevents deploying untested models
- Ensures quality gates are enforced

**Solutions**:
1. **Wait for Dagshub to come back online**
2. **Check Dagshub status**: https://status.dagshub.com (if available)
3. **Verify network connectivity** to Dagshub
4. **Check if it's a temporary outage**

## 🔧 Enhanced Features (New)

### 1. **Connectivity Check**

The script now checks if Dagshub is accessible **before** trying to fetch models:

```python
if not check_dagshub_connectivity(mlflow_tracking_uri):
    print("⚠️ Dagshub appears to be down")
    sys.exit(1)  # Block merge
```

### 2. **Better Logging**

Now shows:
- ✅ Connection status
- ✅ Experiment found/not found
- ✅ Number of runs found
- ✅ Run IDs and timestamps
- ✅ Metrics values
- ✅ Comparison details

### 3. **Duplicate Run Detection**

Checks if latest run is the same as production run:
- If only 1 run exists → Blocks merge (no new model)
- If latest == production → Blocks merge (no new model)

## 📊 What Gets Compared

The script compares these metrics from Dagshub experiments:

| Metric | Source | How It's Fetched |
|--------|--------|------------------|
| **RMSE** | MLflow run metrics | `run.data.metrics.get('rmse')` |
| **MAE** | MLflow run metrics | `run.data.metrics.get('mae')` |
| **R²** | MLflow run metrics | `run.data.metrics.get('r2')` |

These are the **same metrics** that `train.py` logs:
```python
mlflow.log_metric("rmse", rmse)
mlflow.log_metric("mae", mae)
mlflow.log_metric("r2", r2)
```

## 🔄 Workflow

```
1. Training runs (Airflow DAG)
   ↓
2. train.py logs metrics to Dagshub MLflow
   ↓
3. New run appears in "stock_volatility_prediction" experiment
   ↓
4. CI/CD triggers on PR: dev → test
   ↓
5. compare_models.py runs
   ↓
6. Fetches latest 2 runs from Dagshub experiment
   ↓
7. Compares metrics
   ↓
8. Approves/Rejects based on performance
```

## 🐛 Troubleshooting

### Check 1: Is Dagshub accessible?

```bash
curl https://dagshub.com
```

### Check 2: Are runs being logged?

1. Go to your Dagshub repo
2. Click "Experiments" tab
3. Look for experiment: `stock_volatility_prediction`
4. Check if new runs appear after training

### Check 3: Are credentials correct?

```bash
echo $MLFLOW_TRACKING_URI
echo $DAGSHUB_USERNAME
echo $DAGSHUB_TOKEN  # (should show token)
```

### Check 4: Check training logs

Look in Airflow logs for:
- "Model logged to MLflow successfully"
- Any MLflow authentication errors
- Any network errors

## 📝 Summary

| Question | Answer |
|----------|--------|
| **Does it check Dagshub experiments?** | ✅ **YES** - Directly queries Dagshub MLflow |
| **What experiment?** | `stock_volatility_prediction` |
| **What if Dagshub is down?** | ❌ Blocks merge (safety measure) |
| **What if no new runs?** | ❌ Blocks merge (no new model to compare) |
| **How does it authenticate?** | Uses `DAGSHUB_USERNAME` and `DAGSHUB_TOKEN` |

## 🎯 Best Practices

1. **Always check Airflow logs** after training
2. **Verify runs appear in Dagshub** before creating PR
3. **Monitor Dagshub status** if comparisons fail
4. **Keep credentials secure** (use GitHub Secrets)

Your implementation correctly checks Dagshub experiments! 🎉

