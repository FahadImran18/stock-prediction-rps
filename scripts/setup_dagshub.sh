#!/bin/bash
# Setup script for Dagshub integration

echo "Setting up Dagshub integration..."

# Check if .env exists
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "Please edit .env with your credentials"
fi

# Load environment variables
source .env

# Setup DVC remote
echo "Configuring DVC remote..."
dvc remote add -d dagshub "$DVC_REMOTE_URL" || echo "Remote already exists"
dvc remote modify dagshub url "$DVC_REMOTE_URL"
dvc remote modify dagshub user "$DAGSHUB_USERNAME"
dvc remote modify dagshub password "$DAGSHUB_TOKEN"

echo "DVC remote configured!"

# Verify MLflow connection
echo "Verifying MLflow connection..."
python -c "
import os
import mlflow
mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
try:
    mlflow.create_experiment('test')
    print('MLflow connection successful!')
except Exception as e:
    print(f'MLflow connection issue: {e}')
"

echo "Setup complete!"

