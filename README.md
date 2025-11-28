# Real-Time Predictive System (RPS) - Stock Volatility Prediction

This project implements a complete MLOps pipeline for real-time stock volatility prediction using the stockdata.org API.

## Project Overview

The system predicts short-term stock volatility (next hour) using a fully automated MLOps pipeline that includes:
- Automated data ingestion with quality checks
- Feature engineering and data versioning
- Model training and experiment tracking
- CI/CD with automated model comparison
- Production API service with monitoring

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│ stockdata   │────▶│   Airflow    │────▶│   MinIO/S3  │
│    API      │     │     DAG      │     │   Storage   │
└─────────────┘     └──────────────┘     └─────────────┘
                            │
                            ▼
                    ┌──────────────┐
                    │   MLflow     │
                    │  (Dagshub)   │
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

## Project Structure

```
.
├── airflow/
│   ├── dags/
│   │   └── stock_prediction_dag.py
│   └── plugins/
├── src/
│   ├── data/
│   │   ├── extract.py
│   │   ├── transform.py
│   │   ├── quality_check.py
│   │   └── load.py
│   ├── training/
│   │   └── train.py
│   ├── api/
│   │   ├── main.py
│   │   └── models.py
│   └── monitoring/
│       └── metrics.py
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── cd.yml
├── config/
│   └── config.yaml
├── requirements.txt
├── .dvc/
│   └── config
└── README.md
```

## Setup Instructions

### 1. Prerequisites

- Python 3.9+
- Docker and Docker Compose
- Git
- Dagshub account (for MLflow and DVC remote)

### 2. Environment Setup

1. Clone the repository:
```bash
git clone <your-repo-url>
cd Project
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and credentials
```

### 3. Configuration

1. **Dagshub Setup**:
   - Create a repository on Dagshub
   - Get your Dagshub token
   - Update `config/config.yaml` with your Dagshub credentials

2. **Stockdata.org API**:
   - Get your API key from stockdata.org
   - Add it to `.env` file

3. **DVC Setup**:
```bash
dvc remote add -d dagshub <your-dagshub-repo-url>
dvc remote modify dagshub url <your-dagshub-repo-url>
```

4. **MLflow Setup**:
```bash
export MLFLOW_TRACKING_URI=<your-dagshub-mlflow-url>
```

### 4. Airflow Setup

1. Initialize Airflow database:
```bash
airflow db init
```

2. Create Airflow user:
```bash
airflow users create --username admin --firstname Admin --lastname User --role Admin --email admin@example.com --password admin
```

3. Start Airflow:
```bash
airflow webserver --port 8080
airflow scheduler
```

### 5. Run the Pipeline

1. Trigger the Airflow DAG manually or wait for scheduled execution
2. Monitor the DAG execution in Airflow UI (http://localhost:8080)

### 6. Start the API Service

```bash
docker-compose up -d
```

The API will be available at `http://localhost:8000`

### 7. Access Monitoring

- **Grafana**: http://localhost:3000 (default: admin/admin)
- **Prometheus**: http://localhost:9090
- **API Docs**: http://localhost:8000/docs

## CI/CD Workflow

### Branch Strategy
- `dev`: Development branch
- `test`: Testing branch
- `master`: Production branch

### GitHub Actions
- **Feature → dev**: Code quality checks and unit tests
- **dev → test**: Model retraining with CML comparison
- **test → master**: Production deployment

## API Endpoints

- `GET /health`: Health check
- `POST /predict`: Predict stock volatility
- `GET /metrics`: Prometheus metrics

## Monitoring

The system exposes metrics for:
- API inference latency
- Request count
- Data drift detection
- Model performance

## License

MIT License

