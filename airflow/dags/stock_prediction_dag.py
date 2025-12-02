"""
Apache Airflow DAG for Stock Prediction Pipeline
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.standard.operators.bash import BashOperator
import os
import sys
import yaml
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from src.data.extract import extract_stock_data
from src.data.quality_check import validate_data_quality
from src.data.transform import transform_stock_data
from src.data.load import load_to_storage

# Load configuration
config_path = os.path.join(os.path.dirname(__file__), '../../config/config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Default arguments
default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# DAG definition
# Note: days_ago was removed in Airflow 2.4+, using datetime instead
dag = DAG(
    config['airflow']['dag_id'],
    default_args=default_args,
    description='Stock Volatility Prediction Pipeline',
    schedule=config['airflow']['schedule_interval'],  # Changed from schedule_interval to schedule in Airflow 2.4+
    start_date=datetime.now() - timedelta(days=1),  # Replaced days_ago(1)
    catchup=False,
    tags=['mlops', 'stock-prediction'],
)

# Task 1: Extract data
def extract_task(**context):
    """Extract stock data from API"""
    # Try multiple ways to get API key
    # 1. From environment variable
    api_key = os.getenv('STOCKDATA_API_KEY')
    
    # 2. From Airflow Variables
    if not api_key:
        from airflow.models import Variable
        try:
            api_key = Variable.get('STOCKDATA_API_KEY', default_var=None)
        except Exception:
            pass
    
    # 3. From DAG run configuration
    if not api_key and context.get('dag_run') and context.get('dag_run').conf:
        api_key = context.get('dag_run').conf.get('STOCKDATA_API_KEY')
    
    # 4. Try loading from .env file
    if not api_key:
        from pathlib import Path
        env_path = Path(__file__).parent.parent.parent / '.env'
        if env_path.exists():
            load_dotenv(env_path, override=True)
            api_key = os.getenv('STOCKDATA_API_KEY')
    
    if not api_key:
        raise ValueError(
            "STOCKDATA_API_KEY not set. "
            "Options:\n"
            "1. Set in .env file: STOCKDATA_API_KEY=your_key\n"
            "2. Set as Airflow Variable: airflow variables set STOCKDATA_API_KEY your_key\n"
            "3. Pass in DAG run config when triggering"
        )
    
    filepath = extract_stock_data(
        api_key=api_key,
        symbol=config['data']['symbol'],
        interval=config['data']['interval'],
        lookback_days=config['data']['lookback_days'],
        output_path=config['data']['raw_data_path'],
        use_adjusted=config['data'].get('use_adjusted', False)
    )
    
    # Push filepath to XCom for next task
    return filepath

extract = PythonOperator(
    task_id='extract_stock_data',
    python_callable=extract_task,
    dag=dag,
)

# Task 2: Data quality check
def quality_check_task(**context):
    """Validate data quality"""
    # Pull filepath from previous task
    ti = context['ti']
    filepath = ti.xcom_pull(task_ids='extract_stock_data')
    
    validate_data_quality(
        filepath=filepath,
        max_null_percentage=config['quality']['max_null_percentage'],
        required_columns=config['quality']['required_columns'],
        min_rows=config['quality']['min_rows']
    )
    
    return filepath

quality_check = PythonOperator(
    task_id='data_quality_check',
    python_callable=quality_check_task,
    dag=dag,
)

# Task 3: Transform data
def transform_task(**context):
    """Transform and engineer features"""
    ti = context['ti']
    input_filepath = ti.xcom_pull(task_ids='extract_stock_data')
    
    # Generate output path
    from pathlib import Path
    output_dir = Path(config['data']['processed_data_path'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filepath = output_dir / f"processed_data_{timestamp}.parquet"
    
    transform_stock_data(
        input_filepath=input_filepath,
        output_filepath=str(output_filepath)
    )
    
    return str(output_filepath)

transform = PythonOperator(
    task_id='transform_data',
    python_callable=transform_task,
    dag=dag,
)

# Task 4: Generate data profile
def profile_task(**context):
    """Generate data profile report (simple version for Python 3.12 compatibility)"""
    import pandas as pd
    import mlflow
    import json
    
    ti = context['ti']
    filepath = ti.xcom_pull(task_ids='transform_data')
    
    # Load data
    df = pd.read_parquet(filepath)
    
    # Generate simple profile (works with Python 3.12)
    profile_data = {
        "data_rows": len(df),
        "data_columns": len(df.columns),
        "column_names": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": df.isnull().sum().to_dict(),
        "numeric_summary": df.describe().to_dict() if len(df.select_dtypes(include=['number']).columns) > 0 else {},
        "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024**2
    }
    
    # Save profile as JSON
    profile_path = f"data/profiles/profile_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs(os.path.dirname(profile_path), exist_ok=True)
    with open(profile_path, 'w') as f:
        json.dump(profile_data, f, indent=2, default=str)
    
    # Try to use pandas-profiling if available (for older Python versions)
    try:
        from pandas_profiling import ProfileReport
        html_profile = ProfileReport(df, title="Stock Data Profile", minimal=True)
        html_path = profile_path.replace('.json', '.html')
        html_profile.to_file(html_path)
        profile_path = html_path
    except ImportError:
        # pandas-profiling not available (Python 3.12+), use JSON profile
        logging.info("pandas-profiling not available, using JSON profile")
    
    # Log to MLflow
    mlflow_tracking_uri = os.getenv('MLFLOW_TRACKING_URI')
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(config['mlflow']['experiment_name'])
        
        with mlflow.start_run(run_name=f"data_profile_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            mlflow.log_artifact(profile_path, "data_profiles")
            mlflow.log_param("data_rows", len(df))
            mlflow.log_param("data_columns", len(df.columns))
            mlflow.log_param("memory_usage_mb", profile_data["memory_usage_mb"])
    
    return profile_path

profile = PythonOperator(
    task_id='generate_data_profile',
    python_callable=profile_task,
    dag=dag,
)

# Task 5: Load to storage
def load_task(**context):
    """Load processed data to MinIO/S3"""
    ti = context['ti']
    filepath = ti.xcom_pull(task_ids='transform_data')
    
    object_name = load_to_storage(
        filepath=filepath,
        endpoint=config['data']['minio']['endpoint'],
        access_key=config['data']['minio']['access_key'],
        secret_key=config['data']['minio']['secret_key'],
        bucket=config['data']['minio']['bucket'],
        use_minio=True
    )
    
    return object_name

load = PythonOperator(
    task_id='load_to_storage',
    python_callable=load_task,
    dag=dag,
)

# Task 6: DVC versioning
def dvc_version_task(**context):
    """Version data with DVC"""
    ti = context['ti']
    filepath = ti.xcom_pull(task_ids='transform_data')
    
    # Use DVC to add and push data
    import subprocess
    
    # Add file to DVC
    subprocess.run(['dvc', 'add', filepath], check=True)
    
    # Commit DVC metadata
    subprocess.run(['git', 'add', f'{filepath}.dvc'], check=True)
    
    return filepath

dvc_version = PythonOperator(
    task_id='version_data_dvc',
    python_callable=dvc_version_task,
    dag=dag,
)

# Task 7: Train model
def train_task(**context):
    """Train the model"""
    ti = context['ti']
    filepath = ti.xcom_pull(task_ids='transform_data')
    
    # Run training script
    import subprocess
    train_script = os.path.join(os.path.dirname(__file__), '../../src/training/train.py')
    
    result = subprocess.run(
        [sys.executable, train_script, '--data-path', filepath],
        capture_output=True,
        text=True,
        check=True
    )
    
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    
    return "Training completed"

train = PythonOperator(
    task_id='train_model',
    python_callable=train_task,
    dag=dag,
)

# Define task dependencies
extract >> quality_check >> transform >> [profile, load, dvc_version] >> train

