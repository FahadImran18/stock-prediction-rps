#!/bin/bash
# Script to set Airflow Variables for the pipeline

echo "Setting Airflow Variables..."

# Set AIRFLOW_HOME
export AIRFLOW_HOME=$(pwd)/airflow

# Load .env file if it exists
if [ -f .env ]; then
    echo "Loading .env file..."
    export $(cat .env | grep -v '^#' | xargs)
fi

# Set Airflow Variables
echo ""
echo "Setting STOCKDATA_API_KEY..."
if [ -z "$STOCKDATA_API_KEY" ]; then
    echo "ERROR: STOCKDATA_API_KEY not found in .env file"
    echo "Please add it to your .env file first"
    exit 1
fi

airflow variables set STOCKDATA_API_KEY "$STOCKDATA_API_KEY"

# Set MLflow tracking URI if available
if [ ! -z "$MLFLOW_TRACKING_URI" ]; then
    echo "Setting MLFLOW_TRACKING_URI..."
    airflow variables set MLFLOW_TRACKING_URI "$MLFLOW_TRACKING_URI"
fi

echo ""
echo "Variables set successfully!"
echo "Verify with: airflow variables list"

