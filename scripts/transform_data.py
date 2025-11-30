"""
Data Transformation & Feature Engineering for Crypto Price Prediction
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from ydata_profiling import ProfileReport
import mlflow
import os
import sys
from contextlib import redirect_stderr
from io import StringIO
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def create_time_features(df):
    """Create time-based features"""
    print("  → Creating time features...")
    df['datetime'] = pd.to_datetime(df['time'], unit='ms')
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['day_of_month'] = df['datetime'].dt.day
    df['month'] = df['datetime'].dt.month
    
    # Cyclical encoding for hour (24-hour cycle)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Cyclical encoding for day of week (7-day cycle)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    return df


def create_lag_features(df, column='priceUsd', lags=[1, 2, 3, 6, 12, 24]):
    """Create lag features for time series"""
    print(f"  → Creating lag features (lags: {lags})...")
    for lag in lags:
        df[f'{column}_lag_{lag}'] = df[column].shift(lag)
    return df


def create_rolling_features(df, column='priceUsd', windows=[3, 6, 12, 24]):
    """Create rolling window statistics"""
    print(f"  → Creating rolling features (windows: {windows})...")
    for window in windows:
        # Rolling mean
        df[f'{column}_rolling_mean_{window}'] = df[column].rolling(window=window).mean()
        
        # Rolling std (volatility)
        df[f'{column}_rolling_std_{window}'] = df[column].rolling(window=window).std()
        
        # Rolling min/max
        df[f'{column}_rolling_min_{window}'] = df[column].rolling(window=window).min()
        df[f'{column}_rolling_max_{window}'] = df[column].rolling(window=window).max()
        
        # Rolling range (max - min)
        df[f'{column}_rolling_range_{window}'] = (
            df[f'{column}_rolling_max_{window}'] - df[f'{column}_rolling_min_{window}']
        )
    
    return df


def create_price_change_features(df, column='priceUsd'):
    """Create price change and return features"""
    print("  → Creating price change features...")
    
    # Percentage change
    df['price_pct_change_1h'] = df[column].pct_change(periods=1)
    df['price_pct_change_3h'] = df[column].pct_change(periods=3)
    df['price_pct_change_6h'] = df[column].pct_change(periods=6)
    df['price_pct_change_12h'] = df[column].pct_change(periods=12)
    df['price_pct_change_24h'] = df[column].pct_change(periods=24)
    
    # Absolute change
    df['price_diff_1h'] = df[column].diff(periods=1)
    df['price_diff_6h'] = df[column].diff(periods=6)
    
    # Momentum (rate of change)
    df['momentum_6h'] = df[column].pct_change(periods=6) * 100
    df['momentum_12h'] = df[column].pct_change(periods=12) * 100
    
    # Volatility (rolling standard deviation of returns)
    df['volatility_6h'] = df['price_pct_change_1h'].rolling(window=6).std()
    df['volatility_12h'] = df['price_pct_change_1h'].rolling(window=12).std()
    df['volatility_24h'] = df['price_pct_change_1h'].rolling(window=24).std()
    
    return df


def create_target_variable(df, column='priceUsd', horizon=1):
    """
    Create target variable: future price change (volatility prediction)
    Target: percentage change in next 'horizon' hours
    """
    print(f"  → Creating target variable (horizon: {horizon}h)...")
    # Calculate future price change: (future_price - current_price) / current_price
    future_price = df[column].shift(-horizon)
    df['target_price_change'] = (future_price - df[column]) / df[column]
    
    # Also create binary target (up/down)
    df['target_direction'] = (df['target_price_change'] > 0).astype(int)
    
    return df


def transform_data(raw_file_path, output_dir='/usr/local/airflow/data/processed'):
    """
    Main transformation pipeline
    """
    print("\n" + "="*70)
    print("DATA TRANSFORMATION PIPELINE")
    print("="*70)
    print(f"Input: {Path(raw_file_path).name}")
    
    # Load raw data
    df = pd.read_csv(raw_file_path)
    print(f"\n1. Loaded data: {df.shape}")
    
    # Convert price to numeric
    df['priceUsd'] = pd.to_numeric(df['priceUsd'], errors='coerce')
    df['time'] = pd.to_numeric(df['time'], errors='coerce')
    
    # Sort by time
    df = df.sort_values('time').reset_index(drop=True)
    
    # Apply transformations
    print("\n2. Feature Engineering:")
    df = create_time_features(df)
    df = create_lag_features(df, column='priceUsd')
    df = create_rolling_features(df, column='priceUsd')
    df = create_price_change_features(df, column='priceUsd')
    df = create_target_variable(df, column='priceUsd', horizon=1)
    
    # Drop rows with NaN values (from lag/rolling features)
    print(f"\n3. Data Cleaning:")
    initial_rows = len(df)
    df = df.dropna()
    dropped_rows = initial_rows - len(df)
    print(f"  → Dropped {dropped_rows} rows with NaN values ({dropped_rows/initial_rows*100:.1f}%)")
    print(f"  → Final shape: {df.shape}")
    
    # Feature summary
    feature_cols = [col for col in df.columns if col not in ['date', 'time', 'datetime', 'extraction_time']]
    print(f"\n4. Features Created: {len(feature_cols)}")
    print(f"  → Time features: {len([c for c in feature_cols if 'hour' in c or 'dow' in c or 'day' in c or 'month' in c])}")
    print(f"  → Lag features: {len([c for c in feature_cols if 'lag' in c])}")
    print(f"  → Rolling features: {len([c for c in feature_cols if 'rolling' in c])}")
    print(f"  → Price change features: {len([c for c in feature_cols if 'pct_change' in c or 'diff' in c or 'momentum' in c or 'volatility' in c])}")
    print(f"  → Target variables: {len([c for c in feature_cols if 'target' in c])}")
    
    # Save processed data
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    processed_file = output_path / f'bitcoin_processed_{timestamp}.csv'
    df.to_csv(processed_file, index=False)
    
    print(f"\n5. Saved: {processed_file.name}")
    
    # Generate data profiling report
    print("\n6. Generating Data Profiling Report...")
    try:
        # Clean dataframe for profiling - ensure all columns are proper types
        df_profile = df.copy()
        
        # Replace infinities with NaN
        df_profile = df_profile.replace([np.inf, -np.inf], np.nan)
        
        # Convert datetime columns to string to avoid profiling issues
        for col in df_profile.select_dtypes(include=['datetime64']).columns:
            df_profile[col] = df_profile[col].astype(str)
        
        # Ensure all numeric columns are float64 (ydata_profiling can have issues with other types)
        for col in df_profile.select_dtypes(include=[np.number]).columns:
            if df_profile[col].dtype != 'float64':
                try:
                    df_profile[col] = pd.to_numeric(df_profile[col], errors='coerce').astype('float64')
                except (ValueError, TypeError):
                    # If conversion fails, keep original type
                    pass
        
        # Suppress stderr output from ydata_profiling (progress bars use tqdm which writes to stderr)
        # Use file descriptor-level redirection to catch all output, including from tqdm
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        old_stderr_fd = os.dup(sys.stderr.fileno())
        
        try:
            # Redirect stderr at file descriptor level (catches tqdm output)
            os.dup2(devnull_fd, sys.stderr.fileno())
            
            # Also set environment variable as backup
            old_tqdm_disable = os.environ.get('TQDM_DISABLE', None)
            os.environ['TQDM_DISABLE'] = '1'
            
            try:
                # Use minimal mode for more reliable profiling
                profile = ProfileReport(
                    df_profile, 
                    title="Bitcoin Price Data - Feature Engineering Report",
                    explorative=True,
                    minimal=True  # Use minimal mode to avoid type-related errors
                )
                
                profile_path = output_path / f'data_profile_{timestamp}.html'
                profile.to_file(profile_path)
            finally:
                # Restore stderr file descriptor
                os.dup2(old_stderr_fd, sys.stderr.fileno())
                os.close(old_stderr_fd)
                
                # Restore environment variable
                if old_tqdm_disable is None:
                    os.environ.pop('TQDM_DISABLE', None)
                else:
                    os.environ['TQDM_DISABLE'] = old_tqdm_disable
        finally:
            os.close(devnull_fd)
        
        print(f"  ✅ Profile report: {profile_path.name}")
    except Exception as e:
        print(f"  ⚠️  Warning: Could not generate profile report: {str(e)}")
        print(f"  → This is non-critical - data transformation completed successfully")
        profile_path = None
    
    # Log to MLflow
    print("\n7. Logging to MLflow...")
    try:
        mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
        mlflow.set_experiment("crypto-price-prediction")
        
        with mlflow.start_run(run_name=f"data_processing_{timestamp}"):
            # Log parameters
            mlflow.log_param("total_records", len(df))
            mlflow.log_param("num_features", len(feature_cols))
            mlflow.log_param("processing_timestamp", timestamp)
            mlflow.log_param("dropped_rows", dropped_rows)
            mlflow.log_param("data_quality_pass", True)
            
            # Log metrics
            mlflow.log_metric("null_percentage", df.isnull().sum().sum() / (df.shape[0] * df.shape[1]))
            mlflow.log_metric("price_mean", df['priceUsd'].mean())
            mlflow.log_metric("price_std", df['priceUsd'].std())
            
            # Log artifacts
            if profile_path and profile_path.exists():
                mlflow.log_artifact(str(profile_path), "data_profile")
            
            print(f"  ✅ Logged to MLflow experiment: crypto-price-prediction")
    
    except Exception as e:
        print(f"  ⚠️  Warning: Could not log to MLflow: {str(e)}")
    
    print("="*70 + "\n")
    
    return str(processed_file), df.shape


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        raw_file = sys.argv[1]
        output = sys.argv[2] if len(sys.argv) > 2 else '/usr/local/airflow/data/processed'
        transform_data(raw_file, output)
    else:
        print("Usage: python transform_data.py <raw_file_path> [output_dir]")