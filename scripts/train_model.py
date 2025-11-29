"""
Model Training Script with MLflow Experiment Tracking
Predicts Bitcoin price volatility (percentage change)
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import joblib
import os
import warnings
from dotenv import load_dotenv

# Suppress sklearn convergence warnings (often non-critical)
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.linear_model._coordinate_descent')

# ML Libraries
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# MLflow
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

# Load environment
load_dotenv()


class CryptoModelTrainer:
    """Train and evaluate cryptocurrency price prediction models"""
    
    def __init__(self, experiment_name="crypto-price-prediction"):
        """Initialize trainer with MLflow"""
        self.experiment_name = experiment_name
        
        # Set MLflow tracking
        mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
        mlflow.set_experiment(experiment_name)
        
        print(f"✅ MLflow Tracking URI: {os.getenv('MLFLOW_TRACKING_URI')}")
        print(f"✅ Experiment: {experiment_name}")
        
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.scaler = StandardScaler()
    
    def load_data(self, file_path):
        """Load processed data"""
        print(f"\n📂 Loading data from: {Path(file_path).name}")
        
        df = pd.read_csv(file_path)
        print(f"   Shape: {df.shape}")
        
        return df
    
    def prepare_features(self, df, target_col='target_price_change'):
        """Prepare features and target for training"""
        print("\n🔧 Preparing features...")
        
        # Drop non-feature columns
        drop_cols = [
            'date', 'time', 'datetime', 'extraction_time',
            'target_price_change', 'target_direction',
            'priceUsd'  # Don't use current price as feature
        ]
        
        # Get feature columns
        feature_cols = [col for col in df.columns if col not in drop_cols]
        
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # Remove any remaining NaN
        valid_idx = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[valid_idx]
        y = y[valid_idx]
        
        print(f"   Features: {len(feature_cols)}")
        print(f"   Samples: {len(X)}")
        print(f"   Target: {target_col}")
        
        return X, y, feature_cols
    
    def split_data(self, X, y, test_size=0.2, val_size=0.1):
        """Split data into train, validation, and test sets"""
        print("\n✂️  Splitting data...")
        
        # First split: train+val and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=False  # Time series: no shuffle
        )
        
        # Second split: train and val
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=42, shuffle=False
        )
        
        print(f"   Train: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
        print(f"   Val:   {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
        print(f"   Test:  {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def scale_features(self, X_train, X_val, X_test):
        """Scale features using StandardScaler"""
        print("\n📊 Scaling features...")
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        print("   ✓ Features scaled")
        
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def train_model(self, model_name, model, X_train, y_train, X_val, y_val, params=None):
        """Train a single model with MLflow tracking"""
        print(f"\n🤖 Training {model_name}...")
        
        try:
            with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                
                # Log model type
                mlflow.log_param("model_type", model_name)
                mlflow.log_param("timestamp", datetime.now().isoformat())
                
                # Hyperparameter tuning if params provided
                if params:
                    print(f"   Performing GridSearch with {len(params)} param combinations...")
                    grid_search = GridSearchCV(
                        model, params, cv=3, scoring='neg_mean_squared_error',
                        n_jobs=-1, verbose=0
                    )
                    grid_search.fit(X_train, y_train)
                    model = grid_search.best_estimator_
                    
                    print(f"   ✓ Best params: {grid_search.best_params_}")
                    
                    # Log best parameters
                    for param, value in grid_search.best_params_.items():
                        mlflow.log_param(f"best_{param}", value)
                else:
                    # Train with default parameters
                    model.fit(X_train, y_train)
                    
                    # Log default parameters
                    if hasattr(model, 'get_params'):
                        for param, value in model.get_params().items():
                            mlflow.log_param(param, value)
                
                # Predictions
                y_train_pred = model.predict(X_train)
                y_val_pred = model.predict(X_val)
                
                # Calculate metrics
                train_metrics = {
                    'rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
                    'mae': mean_absolute_error(y_train, y_train_pred),
                    'r2': r2_score(y_train, y_train_pred)
                }
                
                val_metrics = {
                    'rmse': np.sqrt(mean_squared_error(y_val, y_val_pred)),
                    'mae': mean_absolute_error(y_val, y_val_pred),
                    'r2': r2_score(y_val, y_val_pred)
                }
                
                # Log metrics to MLflow
                try:
                    mlflow.log_metric("train_rmse", train_metrics['rmse'])
                    mlflow.log_metric("train_mae", train_metrics['mae'])
                    mlflow.log_metric("train_r2", train_metrics['r2'])
                    mlflow.log_metric("val_rmse", val_metrics['rmse'])
                    mlflow.log_metric("val_mae", val_metrics['mae'])
                    mlflow.log_metric("val_r2", val_metrics['r2'])
                except Exception as e:
                    print(f"   ⚠️  Could not log metrics to MLflow: {str(e)}")
                
                # Log model (skip if DagHub doesn't support it)
                try:
                    signature = infer_signature(X_train, y_train_pred)
                    # Use 'name' parameter instead of deprecated 'artifact_path'
                    mlflow.sklearn.log_model(model, name="model", signature=signature)
                    mlflow.sklearn.log_model(self.scaler, name="scaler")
                except Exception as e:
                    if "unsupported endpoint" in str(e):
                        print(f"   ⚠️  DagHub doesn't support model logging - skipping")
                    else:
                        print(f"   ⚠️  Could not log model to MLflow: {str(e)}")
                
                print(f"   ✓ Train RMSE: {train_metrics['rmse']:.6f}")
                print(f"   ✓ Val RMSE:   {val_metrics['rmse']:.6f}")
                print(f"   ✓ Val R²:     {val_metrics['r2']:.4f}")
                
                # Store results
                self.models[model_name] = model
                self.results[model_name] = {
                    'train': train_metrics,
                    'val': val_metrics,
                    'model': model,
                    'run_id': mlflow.active_run().info.run_id if mlflow.active_run() else None
                }
                
                return model, train_metrics, val_metrics
        
        except Exception as e:
            print(f"   ⚠️  MLflow run failed, training locally: {str(e)}")
            
            # Train without MLflow tracking
            if params:
                print(f"   Performing GridSearch with {len(params)} param combinations...")
                grid_search = GridSearchCV(
                    model, params, cv=3, scoring='neg_mean_squared_error',
                    n_jobs=-1, verbose=0
                )
                grid_search.fit(X_train, y_train)
                model = grid_search.best_estimator_
                print(f"   ✓ Best params: {grid_search.best_params_}")
            else:
                model.fit(X_train, y_train)
            
            # Predictions
            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)
            
            # Calculate metrics
            train_metrics = {
                'rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
                'mae': mean_absolute_error(y_train, y_train_pred),
                'r2': r2_score(y_train, y_train_pred)
            }
            
            val_metrics = {
                'rmse': np.sqrt(mean_squared_error(y_val, y_val_pred)),
                'mae': mean_absolute_error(y_val, y_val_pred),
                'r2': r2_score(y_val, y_val_pred)
            }
            
            print(f"   ✓ Train RMSE: {train_metrics['rmse']:.6f}")
            print(f"   ✓ Val RMSE:   {val_metrics['rmse']:.6f}")
            print(f"   ✓ Val R²:     {val_metrics['r2']:.4f}")
            
            # Store results
            self.models[model_name] = model
            self.results[model_name] = {
                'train': train_metrics,
                'val': val_metrics,
                'model': model,
                'run_id': None
            }
            
            return model, train_metrics, val_metrics
    
    def train_all_models(self, X_train, y_train, X_val, y_val):
        """Train multiple models and compare"""
        print("\n" + "="*70)
        print("TRAINING MULTIPLE MODELS")
        print("="*70)
        
        # Define models and hyperparameters (adjusted for small target variance)
        models_config = {
            'RandomForest': {
                'model': RandomForestRegressor(random_state=42, n_jobs=-1),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [5, 8, 10],
                    'min_samples_split': [5, 10],
                    'min_samples_leaf': [2, 4],
                    'max_features': ['sqrt', 'log2']
                }
            },
            'GradientBoosting': {
                'model': GradientBoostingRegressor(random_state=42),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.001, 0.01, 0.05],
                    'max_depth': [2, 3, 4],
                    'min_samples_split': [5, 10],
                    'subsample': [0.8, 0.9]
                }
            },
            'Ridge': {
                'model': Ridge(random_state=42),
                'params': {
                    'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]
                }
            },
            'Lasso': {
                'model': Lasso(random_state=42, max_iter=20000, tol=1e-4),
                'params': {
                    'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1]
                }
            }
        }
        
        # Train each model
        for model_name, config in models_config.items():
            try:
                self.train_model(
                    model_name,
                    config['model'],
                    X_train, y_train,
                    X_val, y_val,
                    config['params']
                )
            except Exception as e:
                print(f"   ❌ Failed to train {model_name}: {str(e)}")
        
        # Find best model
        if not self.results:
            raise ValueError("❌ No models trained successfully!")
        
        best_val_rmse = float('inf')
        for name, result in self.results.items():
            if result['val']['rmse'] < best_val_rmse:
                best_val_rmse = result['val']['rmse']
                self.best_model_name = name
                self.best_model = result['model']
        
        print(f"\n🏆 Best Model: {self.best_model_name}")
        print(f"   Val RMSE: {best_val_rmse:.6f}")
        
        return self.best_model_name, self.best_model
    
    def analyze_feature_importance(self, X_train, feature_cols):
        """Analyze feature importance of best model"""
        print("\n📊 Feature Importance Analysis:")
        
        # Get feature importance if available
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
            indices = np.argsort(importances)[::-1][:10]  # Top 10
            
            print(f"\n   Top 10 Most Important Features ({self.best_model_name}):")
            for i, idx in enumerate(indices, 1):
                print(f"   {i:2d}. {feature_cols[idx]:30s} - {importances[idx]:.6f}")
        
        elif hasattr(self.best_model, 'coef_'):
            coefs = np.abs(self.best_model.coef_)
            indices = np.argsort(coefs)[::-1][:10]  # Top 10
            
            print(f"\n   Top 10 Most Important Features ({self.best_model_name}):")
            for i, idx in enumerate(indices, 1):
                print(f"   {i:2d}. {feature_cols[idx]:30s} - {coefs[idx]:.6f}")
        else:
            print(f"   ⚠️  Feature importance not available for {self.best_model_name}")
    
    def diagnose_model_performance(self, y_train, y_train_pred, y_val, y_val_pred):
        """Diagnose why model performance is poor"""
        print("\n🔍 Model Performance Diagnosis:")
        
        # Calculate statistics
        train_r2 = r2_score(y_train, y_train_pred)
        val_r2 = r2_score(y_val, y_val_pred)
        
        print(f"\n   Train R²: {train_r2:.4f}")
        print(f"   Val R²:   {val_r2:.4f}")
        
        if val_r2 < 0:
            print(f"   ❌ Negative R² indicates model worse than baseline (mean predictor)")
        
        # Check target variance
        target_std = np.std(y_train)
        train_error_std = np.std(y_train - y_train_pred)
        
        print(f"\n   Target std dev:     {target_std:.8f}")
        print(f"   Train error std:    {train_error_std:.8f}")
        
        if train_error_std > target_std * 0.5:
            print(f"   ⚠️  High error relative to target variance - features may not correlate well")
        
        # Check for overfitting
        overfit = val_r2 - train_r2
        if overfit < -0.05:
            print(f"   ⚠️  Model shows signs of overfitting (Val R² - Train R² = {overfit:.4f})")
        elif train_r2 < 0.1 and val_r2 < 0.1:
            print(f"   ⚠️  Poor feature-target correlation - consider feature engineering")
    
        return self.best_model_name, self.best_model
    
    def evaluate_best_model(self, X_test, y_test):
        """Evaluate best model on test set"""
        print("\n" + "="*70)
        print("FINAL EVALUATION ON TEST SET")
        print("="*70)
        
        if self.best_model is None:
            raise ValueError("No model trained yet!")
        
        y_test_pred = self.best_model.predict(X_test)
        
        # Calculate MAPE with protection against division by zero
        # Only calculate MAPE for non-zero values
        non_zero_mask = np.abs(y_test) > 1e-8
        if non_zero_mask.sum() > 0:
            mape = np.mean(np.abs((y_test[non_zero_mask] - y_test_pred[non_zero_mask]) / y_test[non_zero_mask])) * 100
        else:
            mape = np.nan  # Cannot calculate MAPE if all values are zero
        
        test_metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'mae': mean_absolute_error(y_test, y_test_pred),
            'r2': r2_score(y_test, y_test_pred),
            'mape': mape
        }
        
        print(f"\n📊 Test Set Performance ({self.best_model_name}):")
        print(f"   RMSE: {test_metrics['rmse']:.6f}")
        print(f"   MAE:  {test_metrics['mae']:.6f}")
        print(f"   R²:   {test_metrics['r2']:.4f}")
        if np.isnan(test_metrics['mape']):
            print(f"   MAPE: N/A (target values too close to zero)")
        else:
            print(f"   MAPE: {test_metrics['mape']:.2f}%")
        
        # Log to MLflow (skip if endpoint not supported)
        try:
            with mlflow.start_run(run_name=f"best_model_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                mlflow.log_param("model_type", self.best_model_name)
                mlflow.log_param("evaluation", "test_set")
                mlflow.log_metric("test_rmse", test_metrics['rmse'])
                mlflow.log_metric("test_mae", test_metrics['mae'])
                mlflow.log_metric("test_r2", test_metrics['r2'])
                mlflow.log_metric("test_mape", test_metrics['mape'])
        except Exception as e:
            if "unsupported endpoint" in str(e):
                print(f"   ⚠️  DagHub doesn't support test metrics logging - skipping")
            else:
                print(f"   ⚠️  Could not log test metrics to MLflow: {str(e)}")
        
        return test_metrics, y_test_pred
    
    def save_model(self, output_dir='models'):
        """Save best model locally"""
        print(f"\n💾 Saving model...")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save model
        model_file = output_path / f'{self.best_model_name.lower()}_{timestamp}.pkl'
        joblib.dump(self.best_model, model_file)
        print(f"   ✓ Model: {model_file.name}")
        
        # Save scaler
        scaler_file = output_path / f'scaler_{timestamp}.pkl'
        joblib.dump(self.scaler, scaler_file)
        print(f"   ✓ Scaler: {scaler_file.name}")
        
        # Save results
        results_file = output_path / f'training_results_{timestamp}.json'
        results_data = {
            'best_model': self.best_model_name,
            'timestamp': timestamp,
            'models': {
                name: {
                    'train': result['train'],
                    'val': result['val'],
                    'run_id': result['run_id']
                }
                for name, result in self.results.items()
            }
        }
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        print(f"   ✓ Results: {results_file.name}")
        
        return str(model_file), str(scaler_file), str(results_file)
    
    def generate_comparison_report(self, output_dir='models'):
        """Generate model comparison visualization"""
        print(f"\n📈 Generating comparison report...")
        
        if not self.results:
            print("   ⚠️  No results to compare")
            return None
        
        # Prepare data for plotting
        models = list(self.results.keys())
        train_rmse = [self.results[m]['train']['rmse'] for m in models]
        val_rmse = [self.results[m]['val']['rmse'] for m in models]
        val_r2 = [self.results[m]['val']['r2'] for m in models]
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: RMSE comparison
        x = np.arange(len(models))
        width = 0.35
        axes[0].bar(x - width/2, train_rmse, width, label='Train RMSE', alpha=0.8)
        axes[0].bar(x + width/2, val_rmse, width, label='Val RMSE', alpha=0.8)
        axes[0].set_xlabel('Model')
        axes[0].set_ylabel('RMSE')
        axes[0].set_title('Model Performance Comparison (RMSE)')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(models, rotation=45, ha='right')
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)
        
        # Plot 2: R² score
        axes[1].bar(models, val_r2, alpha=0.8, color='green')
        axes[1].set_xlabel('Model')
        axes[1].set_ylabel('R² Score')
        axes[1].set_title('Validation R² Score')
        axes[1].set_xticks(x)  # Set ticks before labels to avoid FixedFormatter warning
        axes[1].set_xticklabels(models, rotation=45, ha='right')
        axes[1].grid(axis='y', alpha=0.3)
        axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        # Save figure
        output_path = Path(output_dir)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_file = output_path / f'model_comparison_{timestamp}.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✓ Report: {plot_file.name}")
        
        # Log to MLflow (skip if endpoint not supported)
        try:
            mlflow.log_artifact(str(plot_file))
        except Exception as e:
            if "unsupported endpoint" in str(e):
                print(f"   ⚠️  DagHub doesn't support artifact logging - skipping")
            else:
                print(f"   ⚠️  Could not log artifact to MLflow: {str(e)}")
        
        return str(plot_file)


def train_pipeline(data_file, output_dir='models'):
    """Complete training pipeline"""
    print("\n" + "="*70)
    print("CRYPTO PRICE PREDICTION - MODEL TRAINING PIPELINE")
    print("="*70)
    print(f"Data: {Path(data_file).name}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    # Initialize trainer
    trainer = CryptoModelTrainer()
    
    # Load data
    df = trainer.load_data(data_file)
    
    # Prepare features
    X, y, feature_cols = trainer.prepare_features(df)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.split_data(X, y)
    
    # Scale features
    X_train_scaled, X_val_scaled, X_test_scaled = trainer.scale_features(
        X_train, X_val, X_test
    )
    
    # Train all models
    best_model_name, best_model = trainer.train_all_models(
        X_train_scaled, y_train, X_val_scaled, y_val
    )
    
    # Evaluate on test set
    test_metrics, y_test_pred = trainer.evaluate_best_model(X_test_scaled, y_test)
    
    # Get predictions for diagnostics
    y_val_pred = trainer.best_model.predict(X_val_scaled)
    y_train_pred = trainer.best_model.predict(X_train_scaled)
    
    # Feature importance analysis
    trainer.analyze_feature_importance(X_train_scaled, feature_cols)
    
    # Model diagnosis
    trainer.diagnose_model_performance(y_train, y_train_pred, y_val, y_val_pred)
    
    # Save model
    model_file, scaler_file, results_file = trainer.save_model(output_dir)
    
    # Generate comparison report
    report_file = trainer.generate_comparison_report(output_dir)
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE")
    print("="*70)
    
    return {
        'best_model': best_model_name,
        'test_metrics': test_metrics,
        'model_file': model_file,
        'scaler_file': scaler_file,
        'results_file': results_file,
        'report_file': report_file,
        'feature_count': len(feature_cols)
    }


if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Train crypto price prediction models')
    parser.add_argument('data_file', type=str, help='Path to processed data CSV file')
    parser.add_argument('--output-dir', type=str, default='models', help='Directory to save models (default: models)')
    
    args = parser.parse_args()
    
    # Verify file exists
    if not Path(args.data_file).exists():
        print(f"❌ Error: File not found: {args.data_file}")
        sys.exit(1)
    
    result = train_pipeline(args.data_file, args.output_dir)
    
    print(f"\n📦 Output Summary:")
    print(f"   Best Model: {result['best_model']}")
    print(f"   Test RMSE: {result['test_metrics']['rmse']:.6f}")
    print(f"   Test R²: {result['test_metrics']['r2']:.4f}")
    print(f"   Model File: {Path(result['model_file']).name}")