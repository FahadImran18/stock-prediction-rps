# Real-Time Predictive System (RPS) - Stock Volatility Prediction

A complete MLOps pipeline for real-time stock volatility prediction using automated data ingestion, model training, CI/CD, and monitoring.

## 🎯 Project Overview

This system predicts short-term stock volatility (next hour) using a fully automated MLOps pipeline that includes:

- **Data Pipeline**: Automated extraction, quality checks, transformation, and versioning
- **Model Training**: MLflow experiment tracking with Dagshub integration
- **Orchestration**: Apache Airflow DAG running every 6 hours
- **CI/CD**: GitHub Actions with automated testing and model comparison (CML)
- **API Service**: FastAPI REST API with Prometheus metrics
- **Monitoring**: Grafana dashboards for real-time metrics and alerts

## 🏗️ Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│ stockdata   │────▶│   Airflow    │────▶│   MinIO/S3  │
│    API      │     │     DAG      │     │   Storage   │
└─────────────┘     └──────────────┘     └─────────────┘
                            │
                            ▼
                    ┌──────────────┐
                    │   MLflow     │
                    │  (Dagshub)  │
                    └──────────────┘
                            │
                            ▼
                    ┌──────────────┐     ┌─────────────┐
                    │  FastAPI     │────▶│ Prometheus  │
                    │   Service    │     │   Metrics   │
                    └──────────────┘     └─────────────┘
                            │
                            ▼
                    ┌──────────────┐
                    │   Grafana    │
                    │  Dashboard   │
                    └──────────────┘
```

## 📁 Project Structure

```
.
├── airflow/dags/          # Airflow DAG definitions
├── src/
│   ├── data/             # ETL pipeline (extract, transform, load)
│   ├── training/         # Model training script
│   ├── api/              # FastAPI service
│   └── monitoring/       # Prometheus metrics
├── docker/               # Docker configurations
├── .github/workflows/    # CI/CD pipelines
├── config/               # Configuration files
└── scripts/              # Utility scripts
```

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- Docker & Docker Compose
- Git
- Dagshub account
- Stockdata.org API key

### Setup Steps

1. **Clone and setup environment:**
   ```bash
   git clone <repo-url>
   cd Project
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and credentials
   ```

3. **Initialize Airflow:**
   ```bash
   export AIRFLOW_HOME=$(pwd)/airflow
   airflow db init
   airflow users create --username admin --password admin --role Admin --email admin@example.com
   ```

4. **Start services:**
   ```bash
   cd docker
   docker-compose up -d
   ```

5. **Start Airflow:**
   ```bash
   airflow webserver --port 8080  # Terminal 1
   airflow scheduler               # Terminal 2
   ```

6. **Access services:**
   - Airflow UI: http://localhost:8080 (admin/admin)
   - API Docs: http://localhost:8000/docs
   - Grafana: http://localhost:3000 (admin/admin)
   - Prometheus: http://localhost:9090

## 📚 Documentation

- **[TEAM_SETUP.md](TEAM_SETUP.md)** - Complete setup guide for team members
- **[CI_CD_SETUP.md](CI_CD_SETUP.md)** - GitHub Actions CI/CD configuration
- **[GRAFANA_SETUP.md](GRAFANA_SETUP.md)** - Grafana dashboard and alerting setup
- **[SETUP.md](SETUP.md)** - Detailed technical setup (advanced)

## 🔧 Key Features

### Data Pipeline
- **Extraction**: Fetches stock data from stockdata.org API
- **Quality Checks**: Validates data quality (nulls, schema, min rows)
- **Transformation**: Feature engineering (lag features, rolling means, volatility)
- **Versioning**: DVC for data version control
- **Storage**: Optional MinIO/S3 storage

### Model Management
- **Training**: Gradient Boosting Regressor with hyperparameter tracking
- **Tracking**: MLflow experiment tracking via Dagshub
- **Metrics**: RMSE, MAE, R², MAPE logged to MLflow
- **Artifacts**: Models and feature columns stored in MLflow

### CI/CD Pipeline
- **Feature → dev**: Code quality checks and unit tests
- **dev → test**: Model retraining with CML comparison reports
- **test → master**: Docker image build and deployment verification

### Monitoring
- **Metrics**: API latency, request count, inference time, data drift
- **Dashboards**: Real-time Grafana visualizations
- **Alerts**: Configurable alerts for latency and drift thresholds

## 🛠️ Technology Stack

| Category | Tools |
|----------|-------|
| **Orchestration** | Apache Airflow |
| **Data/Model Mgmt** | DVC, MLflow, Dagshub |
| **CI/CD** | GitHub Actions, CML |
| **API** | FastAPI |
| **Monitoring** | Prometheus, Grafana |
| **Storage** | MinIO (S3-compatible) |
| **Containerization** | Docker, Docker Compose |

## 📊 API Endpoints

- `GET /health` - Health check
- `POST /predict` - Predict stock volatility
- `GET /metrics` - Prometheus metrics

## 🔐 Required Credentials

Set these in `.env` or as Airflow Variables:

- `STOCKDATA_API_KEY` - Stockdata.org API key
- `MLFLOW_TRACKING_URI` - Dagshub MLflow URI
- `DAGSHUB_USERNAME` - Dagshub username
- `DAGSHUB_TOKEN` - Dagshub access token

## 📝 License

MIT License

## 🤝 Contributing

See [TEAM_SETUP.md](TEAM_SETUP.md) for development setup instructions.
