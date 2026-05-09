# Real-Time Predictive System (RPS)

Stock volatility prediction with a full MLOps pipeline — data ingestion, model training, CI/CD, and live monitoring.

## What this does

RPS predicts short-term stock volatility (next-hour) by pulling live market data, running it through a feature engineering pipeline, and serving predictions via a REST API. The whole thing is automated: new data comes in every 6 hours, models get retrained and compared against the current champion, and metrics are visible in Grafana.

## Architecture
stockdata API  →  Airflow DAG  →  MinIO/S3
|
MLflow (Dagshub)
|
FastAPI  →  Prometheus  →  Grafana

## Project layout
.
├── airflow/dags/      # DAG definitions
├── src/
│   ├── data/          # Extract, transform, load
│   ├── training/      # Model training
│   ├── api/           # FastAPI service
│   └── monitoring/    # Prometheus metrics
├── docker/
├── .github/workflows/
├── config/
└── scripts/

## Getting started

You'll need Python 3.12+, Docker, and accounts on Dagshub and stockdata.org.

```bash
git clone <repo-url>
cd Project
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Fill in your API keys

export AIRFLOW_HOME=$(pwd)/airflow
airflow db init
airflow users create --username admin --password admin --role Admin --email admin@example.com

cd docker && docker-compose up -d

# In separate terminals:
airflow webserver --port 8080
airflow scheduler
```

Services run at:

- Airflow: http://localhost:8080 (admin/admin)
- API docs: http://localhost:8000/docs
- Grafana: http://localhost:3000 (admin/admin)
- Prometheus: http://localhost:9090

## How the pipeline works

**Data** — Fetches OHLCV data from stockdata.org, validates schema and row counts, engineers lag features and rolling volatility windows, then versions everything with DVC.

**Training** — Trains a Gradient Boosting Regressor and logs RMSE, MAE, R², and MAPE to MLflow via Dagshub. Model artifacts and feature columns are stored as MLflow artifacts.

**CI/CD** — Three-stage GitHub Actions pipeline: feature branches run linting and unit tests; merges to `dev` retrain the model and generate a CML comparison report against the current production model; merges to `master` build and verify the Docker image.

**Monitoring** — Prometheus scrapes API latency, request volume, inference time, and data drift metrics. Grafana dashboards surface these with configurable alert thresholds.

## API
GET  /health    Health check
POST /predict   Predict next-hour volatility for a given ticker
GET  /metrics   Prometheus metrics endpoint

## Required credentials

Set these in `.env` or as Airflow Variables:

| Variable | Description |
|---|---|
| `STOCKDATA_API_KEY` | stockdata.org API key |
| `MLFLOW_TRACKING_URI` | Dagshub MLflow tracking URI |
| `DAGSHUB_USERNAME` | Dagshub username |
| `DAGSHUB_TOKEN` | Dagshub access token |

## Stack

Airflow, MLflow, DVC, Dagshub, FastAPI, Prometheus, Grafana, MinIO, Docker, GitHub Actions, CML.

## License

MIT
