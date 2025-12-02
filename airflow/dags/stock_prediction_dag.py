"""
Apache Airflow DAG for Stock Prediction Pipeline
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.standard.operators.bash import BashOperator
from airflow.models import Variable
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
    
    # Log to MLflow (optional - skip if not configured or auth fails)
    mlflow_tracking_uri = os.getenv('MLFLOW_TRACKING_URI')
    if not mlflow_tracking_uri:
        try:
            mlflow_tracking_uri = Variable.get('MLFLOW_TRACKING_URI', default_var=None)
        except Exception:
            pass
    
    if mlflow_tracking_uri:
        try:
            # Try to get Dagshub credentials from Airflow Variables or environment
            dagshub_username = os.getenv('DAGSHUB_USERNAME')
            dagshub_token = os.getenv('DAGSHUB_TOKEN')
            
            if not dagshub_username:
                try:
                    dagshub_username = Variable.get('DAGSHUB_USERNAME', default_var=None)
                except Exception:
                    pass
            
            if not dagshub_token:
                try:
                    dagshub_token = Variable.get('DAGSHUB_TOKEN', default_var=None)
                except Exception:
                    pass
            
            # For Dagshub MLflow, embed credentials in URI if available
            if dagshub_username and dagshub_token and 'dagshub.com' in mlflow_tracking_uri:
                # Embed credentials in URI: https://username:token@dagshub.com/...
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
            
            # Try to get or create experiment
            try:
                # Try to get existing experiment
                experiment = mlflow.get_experiment_by_name(config['mlflow']['experiment_name'])
                if experiment is None:
                    # Create if doesn't exist
                    experiment_id = mlflow.create_experiment(config['mlflow']['experiment_name'])
                    logging.info(f"Created MLflow experiment: {experiment_id}")
                mlflow.set_experiment(config['mlflow']['experiment_name'])
            except Exception as exp_error:
                logging.warning(f"Could not set/create MLflow experiment: {exp_error}")
                logging.info("Skipping MLflow logging - check credentials and permissions")
                return profile_path
            
            # Log to MLflow
            with mlflow.start_run(run_name=f"data_profile_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                mlflow.log_artifact(profile_path, "data_profiles")
                mlflow.log_param("data_rows", len(df))
                mlflow.log_param("data_columns", len(df.columns))
                mlflow.log_param("memory_usage_mb", profile_data["memory_usage_mb"])
                logging.info("Successfully logged profile to MLflow")
        except Exception as e:
            logging.warning(f"MLflow logging failed: {e}")
            logging.info("Continuing without MLflow logging - profile saved locally")
            import traceback
            logging.debug(traceback.format_exc())
    else:
        logging.info("MLFLOW_TRACKING_URI not set, skipping MLflow logging")
    
    return profile_path

profile = PythonOperator(
    task_id='generate_data_profile',
    python_callable=profile_task,
    dag=dag,
)

# Task 5: Load to storage
def load_task(**context):
    """Load processed data to MinIO/S3 (optional - skips if MinIO unavailable)"""
    import socket
    import logging
    
    ti = context['ti']
    filepath = ti.xcom_pull(task_ids='transform_data')
    
    # Check if MinIO is available
    endpoint = config['data']['minio']['endpoint']
    host, port = endpoint.split(':') if ':' in endpoint else (endpoint, '9000')
    
    try:
        # Try to connect to MinIO
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex((host, int(port)))
        sock.close()
        
        if result != 0:
            logging.warning(f"MinIO not available at {endpoint}, skipping storage upload")
            logging.info("To enable storage, start MinIO: docker-compose up -d minio")
            return None
    except Exception as e:
        logging.warning(f"Could not check MinIO availability: {e}, skipping storage upload")
        return None
    
    # MinIO is available, try to upload
    try:
        object_name = load_to_storage(
            filepath=filepath,
            endpoint=endpoint,
            access_key=config['data']['minio']['access_key'],
            secret_key=config['data']['minio']['secret_key'],
            bucket=config['data']['minio']['bucket'],
            use_minio=True
        )
        logging.info(f"Successfully uploaded to MinIO: {object_name}")
        return object_name
    except Exception as e:
        logging.warning(f"Failed to upload to MinIO: {e}")
        logging.info("Continuing pipeline without storage upload")
        return None

load = PythonOperator(
    task_id='load_to_storage',
    python_callable=load_task,
    dag=dag,
)

# Task 6: DVC versioning
def dvc_version_task(**context):
    """Version data with DVC"""
    import subprocess
    import logging
    
    ti = context['ti']
    filepath = ti.xcom_pull(task_ids='transform_data')
    
    if not filepath or not os.path.exists(filepath):
        raise ValueError(f"Data file not found: {filepath}")
    
    try:
        # Add file to DVC (this creates .dvc file and .gitignore entry)
        logging.info(f"Adding {filepath} to DVC...")
        result = subprocess.run(
            ['dvc', 'add', filepath],
            check=True,
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        )
        logging.info(f"DVC add output: {result.stdout}")
        
        # The .dvc file should be created
        dvc_file = f'{filepath}.dvc'
        if os.path.exists(dvc_file):
            logging.info(f"DVC file created: {dvc_file}")
        else:
            logging.warning(f"DVC file not found: {dvc_file}")
        
        # Note: Git commit should be done separately or via git hook
        # We'll just add the .dvc file to git staging
        try:
            subprocess.run(
                ['git', 'add', dvc_file],
                check=False,  # Don't fail if git not initialized
                capture_output=True,
                text=True,
                cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            )
        except Exception as e:
            logging.warning(f"Could not add .dvc file to git: {e}")
        
        return filepath
        
    except subprocess.CalledProcessError as e:
        logging.error(f"DVC command failed: {e}")
        logging.error(f"STDERR: {e.stderr}")
        logging.error(f"STDOUT: {e.stdout}")
        raise

dvc_version = PythonOperator(
    task_id='version_data_dvc',
    python_callable=dvc_version_task,
    dag=dag,
)

# Task 7: Train model
def train_task(**context):
    """Train the model"""
    import subprocess
    
    ti = context['ti']
    filepath = ti.xcom_pull(task_ids='transform_data')
    
    if not filepath:
        raise ValueError("No data file path received from transform_data task")
    
    # Get MLflow and Dagshub credentials from environment or Airflow Variables
    env = os.environ.copy()
    
    # MLflow tracking URI
    mlflow_uri = (
        os.getenv('MLFLOW_TRACKING_URI') or
        Variable.get('MLFLOW_TRACKING_URI', default_var=None)
    )
    if mlflow_uri:
        env['MLFLOW_TRACKING_URI'] = mlflow_uri
    
    # Dagshub credentials
    dagshub_username = (
        os.getenv('DAGSHUB_USERNAME') or
        Variable.get('DAGSHUB_USERNAME', default_var=None)
    )
    if dagshub_username:
        env['DAGSHUB_USERNAME'] = dagshub_username
    
    dagshub_token = (
        os.getenv('DAGSHUB_TOKEN') or
        Variable.get('DAGSHUB_TOKEN', default_var=None)
    )
    if dagshub_token:
        env['DAGSHUB_TOKEN'] = dagshub_token
    
    # Run training script
    train_script = os.path.join(os.path.dirname(__file__), '../../src/training/train.py')
    
    # Change to project root directory for relative paths to work
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    
    logging.info(f"Running training script: {train_script}")
    logging.info(f"Data file: {filepath}")
    logging.info(f"Working directory: {project_root}")
    
    result = subprocess.run(
        [sys.executable, train_script, '--data-path', filepath],
        cwd=project_root,
        env=env,
        capture_output=True,
        text=True,
        check=False  # Don't raise immediately, log errors first
    )
    
    # Log output
    if result.stdout:
        logging.info(f"Training stdout:\n{result.stdout}")
    if result.stderr:
        logging.warning(f"Training stderr:\n{result.stderr}")
    
    # Check if training succeeded
    if result.returncode != 0:
        error_msg = f"Training failed with exit code {result.returncode}"
        if result.stderr:
            error_msg += f"\nError: {result.stderr}"
        logging.error(error_msg)
        raise RuntimeError(error_msg)
    
    logging.info("Training completed successfully")
    return "Training completed"

train = PythonOperator(
    task_id='train_model',
    python_callable=train_task,
    dag=dag,
)

# Define task dependencies
extract >> quality_check >> transform >> [profile, load, dvc_version] >> train

