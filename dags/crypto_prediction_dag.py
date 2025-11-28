"""
Crypto Price Prediction - MLOps Pipeline DAG
Complete ETL + Feature Engineering + Storage + Versioning
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.exceptions import AirflowException
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd
import json
from pathlib import Path
import sys
import os
from airflow.models import Variable

# Add scripts to path
sys.path.append('/usr/local/airflow/scripts')

# CoinCap API key: prefer Airflow Variable `coincap_api_key`, then env `COINCAP_API_KEY`,
# otherwise fall back to the provided literal (not recommended for production).
COINCAP_API_KEY = Variable.get(
    'coincap_api_key',
    default_var=os.environ.get(
        'COINCAP_API_KEY',
        'cef44a548998e044fb736718d6299437d7a1740c9ca2f3786bb9569050c6787d',
    ),
)

# Default arguments
default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

# Initialize DAG
dag = DAG(
    'crypto_price_prediction_pipeline',
    default_args=default_args,
    description='End-to-end MLOps pipeline for crypto price prediction with DagHub',
    schedule='0 */6 * * *',  # Run every 6 hours
    catchup=False,
    tags=['mlops', 'crypto', 'prediction', 'phase-1'],
)


def extract_crypto_data(**context):
    """
    Task 1: Extract cryptocurrency data from CoinCap API
    """
    print("\n" + "="*70)
    print("TASK 1: DATA EXTRACTION")
    print("="*70)
    print("Source: CoinCap API")
    print("Asset: Bitcoin (BTC)")
    print("Interval: Hourly (h1)")
    
    # API endpoint for Bitcoin hourly data
    url = "https://rest.coincap.io/v3/assets/bitcoin/history"
    params = {
        'interval': 'h1',  # 1 hour intervals
    }
    
    try:
        print("\nFetching data from API...")
        # Create raw data directory early so we can fallback to cached files
        raw_data_dir = Path('/usr/local/airflow/data/raw')
        raw_data_dir.mkdir(parents=True, exist_ok=True)

        # session with retries to handle transient network errors
        session = requests.Session()
        retries = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET"]) 
        adapter = HTTPAdapter(max_retries=retries)
        session.mount("https://", adapter)
        session.mount("http://", adapter)

        # Include API key in Authorization header when provided
        headers = {'Authorization': f'Bearer {COINCAP_API_KEY}'} if COINCAP_API_KEY else None
        response = session.get(url, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()

        if 'data' not in data:
            raise AirflowException("Invalid API response: 'data' key not found")

        # Convert to DataFrame
        df = pd.DataFrame(data['data'])

        # Add extraction timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        df['extraction_time'] = datetime.now().isoformat()

        # Save raw data with timestamp
        raw_file_path = raw_data_dir / f'bitcoin_raw_{timestamp}.csv'
        df.to_csv(raw_file_path, index=False)

        print(f"\n✅ Extraction Complete")
        print(f"  → Records extracted: {len(df)}")
        print(f"  → Columns: {list(df.columns)}")
        print(f"  → File saved: {raw_file_path.name}")
        print(f"  → File size: {raw_file_path.stat().st_size / 1024:.2f} KB")
        print("="*70 + "\n")

        # Push file path to XCom for next task
        context['task_instance'].xcom_push(key='raw_file_path', value=str(raw_file_path))
        context['task_instance'].xcom_push(key='record_count', value=len(df))
        context['task_instance'].xcom_push(key='extraction_timestamp', value=timestamp)

        return str(raw_file_path)

    except requests.exceptions.RequestException as e:
        # Attempt to fallback to the most recent cached raw file if network/DNS fails
        print(f"Network request failed: {e}. Attempting to use cached data if available...")
        cached_files = sorted(raw_data_dir.glob('bitcoin_raw_*.csv'), key=lambda p: p.stat().st_mtime, reverse=True)
        if cached_files:
            latest = cached_files[0]
            df = pd.read_csv(latest)
            context['task_instance'].xcom_push(key='raw_file_path', value=str(latest))
            context['task_instance'].xcom_push(key='record_count', value=len(df))
            context['task_instance'].xcom_push(key='extraction_timestamp', value=latest.stem.split('_')[-1])
            print(f"Using cached file: {latest.name} (records: {len(df)})")
            return str(latest)
        else:
            raise AirflowException(f"API request failed and no cached data available: {str(e)}")
    except Exception as e:
        raise AirflowException(f"Data extraction failed: {str(e)}")


def validate_data_quality(**context):
    """
    Task 2: Mandatory Data Quality Gate
    Checks for null values, schema validation, and data completeness
    """
    print("\n" + "="*70)
    print("TASK 2: DATA QUALITY VALIDATION")
    print("="*70)
    
    # Get file path from previous task
    raw_file_path = context['task_instance'].xcom_pull(
        task_ids='extract_data', 
        key='raw_file_path'
    )
    
    if not raw_file_path or not Path(raw_file_path).exists():
        raise AirflowException(f"Raw data file not found: {raw_file_path}")
    
    print(f"Validating: {Path(raw_file_path).name}")
    
    # Load data
    df = pd.read_csv(raw_file_path)
    
    # Quality checks
    quality_checks = {
        'file': Path(raw_file_path).name,
        'timestamp': datetime.now().isoformat(),
        'total_records': len(df),
        'total_columns': len(df.columns),
        'null_checks': {},
        'schema_validation': {},
        'data_validation': {},
        'status': 'PASSED'
    }
    
    print("\n1. Schema Validation:")
    # 1. Check for required columns
    required_columns = ['priceUsd', 'time', 'date']
    for col in required_columns:
        if col not in df.columns:
            quality_checks['status'] = 'FAILED'
            quality_checks['schema_validation'][col] = 'MISSING'
            raise AirflowException(f"❌ FAILED: Required column missing: {col}")
        else:
            print(f"  ✓ Column '{col}' present")
            quality_checks['schema_validation'][col] = 'PRESENT'
    
    print("\n2. Null Value Checks:")
    # 2. Check for null values (fail if >1% nulls in key columns)
    null_threshold = 0.01  # 1%
    
    for col in required_columns:
        null_count = df[col].isnull().sum()
        null_percentage = null_count / len(df)
        
        quality_checks['null_checks'][col] = {
            'null_count': int(null_count),
            'null_percentage': float(null_percentage),
            'threshold': null_threshold,
            'status': 'PASS' if null_percentage <= null_threshold else 'FAIL'
        }
        
        print(f"  → {col}: {null_count} nulls ({null_percentage*100:.2f}%)", end="")
        
        if null_percentage > null_threshold:
            quality_checks['status'] = 'FAILED'
            print(" ❌ FAILED")
            raise AirflowException(
                f"❌ FAILED: {col} has {null_percentage*100:.2f}% null values (threshold: {null_threshold*100}%)"
            )
        else:
            print(" ✓ PASS")
    
    print("\n3. Data Type Validation:")
    # 3. Validate data types
    expected_types = {
        'priceUsd': ['object', 'float64'],
        'time': ['int64', 'object'],
        'date': ['object']
    }
    
    for col, valid_types in expected_types.items():
        actual_type = str(df[col].dtype)
        quality_checks['schema_validation'][f'{col}_dtype'] = actual_type
        
        if actual_type in valid_types:
            print(f"  ✓ {col}: {actual_type}")
        else:
            print(f"  ⚠️  {col}: {actual_type} (expected: {valid_types})")
    
    print("\n4. Data Completeness:")
    # 4. Check for minimum record count
    min_records = 10
    quality_checks['data_validation']['min_records_required'] = min_records
    quality_checks['data_validation']['records_present'] = len(df)
    
    if len(df) < min_records:
        quality_checks['status'] = 'FAILED'
        print(f"  ❌ FAILED: Only {len(df)} records (minimum: {min_records})")
        raise AirflowException(
            f"❌ FAILED: Insufficient data: {len(df)} records (minimum: {min_records})"
        )
    else:
        print(f"  ✓ PASS: {len(df)} records (minimum: {min_records})")
    
    print("\n5. Value Validation:")
    # 5. Validate price values are positive
    df['priceUsd'] = pd.to_numeric(df['priceUsd'], errors='coerce')
    invalid_prices = (df['priceUsd'] <= 0).sum() + df['priceUsd'].isnull().sum()
    
    quality_checks['data_validation']['invalid_prices'] = int(invalid_prices)
    
    if invalid_prices > 0:
        quality_checks['status'] = 'FAILED'
        print(f"  ❌ FAILED: {invalid_prices} invalid price values")
        raise AirflowException("❌ FAILED: Invalid price values detected (negative or null)")
    else:
        print(f"  ✓ PASS: All prices valid (min: ${df['priceUsd'].min():.2f}, max: ${df['priceUsd'].max():.2f})")
    
    # Save quality report
    quality_report_path = Path(raw_file_path).parent / f'quality_report_{Path(raw_file_path).stem}.json'
    with open(quality_report_path, 'w') as f:
        json.dump(quality_checks, f, indent=2)
    
    print(f"\n✅ Data Quality Validation: {quality_checks['status']}")
    print(f"  → Quality report saved: {quality_report_path.name}")
    print("="*70 + "\n")
    
    context['task_instance'].xcom_push(key='quality_report', value=quality_checks)
    context['task_instance'].xcom_push(key='quality_report_path', value=str(quality_report_path))
    
    return quality_checks


def transform_crypto_data(**context):
    """
    Task 3: Transform and engineer features for crypto data
    """
    from transform_data import transform_data
    
    print("\n" + "="*70)
    print("TASK 3: DATA TRANSFORMATION & FEATURE ENGINEERING")
    print("="*70)
    
    # Get file path from extraction task
    raw_file_path = context['task_instance'].xcom_pull(
        task_ids='extract_data',
        key='raw_file_path'
    )
    
    # Transform data
    processed_file, data_shape = transform_data(raw_file_path)
    
    print(f"✅ Transformation Complete")
    print(f"  → Output shape: {data_shape}")
    print(f"  → File: {Path(processed_file).name}")
    
    # Push to XCom
    context['task_instance'].xcom_push(key='processed_file_path', value=processed_file)
    context['task_instance'].xcom_push(key='data_shape', value=str(data_shape))
    
    return processed_file


def upload_to_storage(**context):
    """
    Task 4: Upload processed data to DagHub Storage and version with DVC
    """
    from upload_to_storage import upload_and_version_data
    
    print("\n" + "="*70)
    print("TASK 4: STORAGE & VERSIONING")
    print("="*70)
    
    # Get processed file path
    processed_file_path = context['task_instance'].xcom_pull(
        task_ids='transform_data',
        key='processed_file_path'
    )
    
    # Upload and version
    result = upload_and_version_data(processed_file_path)
    
    # Push results to XCom
    context['task_instance'].xcom_push(key='storage_result', value=result)
    
    return result


def pipeline_summary(**context):
    """
    Task 5: Generate pipeline execution summary
    """
    print("\n" + "="*70)
    print("PIPELINE EXECUTION SUMMARY")
    print("="*70)
    
    # Gather all XCom data
    extraction_data = {
        'timestamp': context['task_instance'].xcom_pull(task_ids='extract_data', key='extraction_timestamp'),
        'record_count': context['task_instance'].xcom_pull(task_ids='extract_data', key='record_count'),
        'file_path': context['task_instance'].xcom_pull(task_ids='extract_data', key='raw_file_path')
    }
    
    quality_data = context['task_instance'].xcom_pull(task_ids='validate_data_quality', key='quality_report')
    
    transform_data = {
        'file_path': context['task_instance'].xcom_pull(task_ids='transform_data', key='processed_file_path'),
        'shape': context['task_instance'].xcom_pull(task_ids='transform_data', key='data_shape')
    }
    
    storage_data = context['task_instance'].xcom_pull(task_ids='upload_to_storage', key='storage_result')
    
    # Print summary
    print(f"\n📊 Execution Time: {datetime.now().isoformat()}")
    print(f"\n1. Data Extraction:")
    print(f"   ✓ Records: {extraction_data['record_count']}")
    print(f"   ✓ File: {Path(extraction_data['file_path']).name}")
    
    print(f"\n2. Quality Validation:")
    print(f"   ✓ Status: {quality_data['status']}")
    print(f"   ✓ Total Records: {quality_data['total_records']}")
    
    print(f"\n3. Transformation:")
    print(f"   ✓ Output Shape: {transform_data['shape']}")
    print(f"   ✓ File: {Path(transform_data['file_path']).name}")
    
    print(f"\n4. Storage & Versioning:")
    print(f"   ✓ Local: {Path(storage_data['local_path']).name}")
    print(f"   ✓ DVC: {Path(storage_data['dvc_file']).name if storage_data['dvc_file'] else 'Not versioned'}")
    print(f"   ✓ Backend: {storage_data['storage_backend']}")
    
    print("\n" + "="*70)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY")
    print("="*70 + "\n")
    
    summary = {
        'pipeline': 'crypto_price_prediction',
        'execution_time': datetime.now().isoformat(),
        'status': 'SUCCESS',
        'extraction': extraction_data,
        'quality': quality_data,
        'transformation': transform_data,
        'storage': storage_data
    }
    
    # Save summary
    summary_path = Path('/usr/local/airflow/logs') / f'pipeline_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary


# Define tasks
extract_task = PythonOperator(
    task_id='extract_data',
    python_callable=extract_crypto_data,
    dag=dag,
)

validate_task = PythonOperator(
    task_id='validate_data_quality',
    python_callable=validate_data_quality,
    dag=dag,
)

transform_task = PythonOperator(
    task_id='transform_data',
    python_callable=transform_crypto_data,
    dag=dag,
)

storage_task = PythonOperator(
    task_id='upload_to_storage',
    python_callable=upload_to_storage,
    dag=dag,
)

summary_task = PythonOperator(
    task_id='pipeline_summary',
    python_callable=pipeline_summary,
    dag=dag,
)

# Set task dependencies
extract_task >> validate_task >> transform_task >> storage_task >> summary_task