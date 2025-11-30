"""
FastAPI Application for Crypto Price Prediction
Serves ML model with REST API
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import os
from datetime import datetime

# Prometheus monitoring
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from prometheus_fastapi_instrumentator import Instrumentator
from starlette.responses import Response

# Initialize FastAPI app
app = FastAPI(
    title="Crypto Price Prediction API",
    description="ML model for predicting Bitcoin price volatility",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics
prediction_counter = Counter(
    'predictions_total',
    'Total number of predictions made'
)

prediction_histogram = Histogram(
    'prediction_duration_seconds',
    'Time spent processing prediction'
)

data_drift_gauge = Gauge(
    'data_drift_ratio',
    'Ratio of out-of-distribution features'
)

model_version_info = Gauge(
    'model_version_info',
    'Model version information',
    ['model_name', 'version']
)

# Global variables for model
model = None
scaler = None
feature_names = None
model_metadata = {}

# Feature statistics for drift detection (set during model loading)
feature_stats = {}


class PredictionRequest(BaseModel):
    """Request model for predictions"""
    features: Dict[str, float] = Field(
        ...,
        description="Dictionary of feature names and values",
        example={
            "hour_sin": 0.5,
            "hour_cos": 0.866,
            "priceUsd_lag_1": 45000.0,
            "priceUsd_rolling_mean_6": 44800.0,
            "price_pct_change_1h": 0.002,
            "volatility_6h": 0.015
        }
    )
    timestamp: Optional[str] = Field(
        None,
        description="Optional timestamp (ISO format) for computing time features. If not provided, time features must be included in features dict."
    )


class PredictionResponse(BaseModel):
    """Response model for predictions"""
    prediction: float = Field(..., description="Predicted price change percentage")
    model_version: str = Field(..., description="Model version used")
    timestamp: str = Field(..., description="Prediction timestamp")
    confidence: Optional[str] = Field(None, description="Confidence level")
    drift_detected: bool = Field(..., description="Whether data drift was detected")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    model_version: str
    timestamp: str


class ModelInfoResponse(BaseModel):
    """Model information response"""
    model_type: str
    version: str
    features_count: int
    loaded_at: str
    metadata: Dict


def load_model(model_path: str = "models", model_name: str = None):
    """Load the trained model and scaler"""
    global model, scaler, feature_names, model_metadata, feature_stats
    
    # Resolve path relative to project root (parent of app directory)
    if not Path(model_path).is_absolute():
        # Get the project root (parent of app directory)
        project_root = Path(__file__).parent.parent
        model_dir = project_root / model_path
    else:
        model_dir = Path(model_path)
    
    # Find latest model if not specified
    if model_name is None:
        # Get all .pkl files but exclude scaler files
        all_pkl_files = sorted(model_dir.glob("*.pkl"))
        model_files = [f for f in all_pkl_files if not f.name.startswith("scaler_")]
        if not model_files:
            raise FileNotFoundError(f"No model files found in {model_path}")
        model_file = model_files[-1]  # Get latest
    else:
        model_file = model_dir / model_name
    
    # Find corresponding scaler - match by timestamp in filename
    if model_name is None:
        # Extract timestamp from model filename (e.g., ridge_20251129_165545.pkl -> 20251129_165545)
        model_stem = model_file.stem  # e.g., "ridge_20251129_165545"
        # Try to extract timestamp part (everything after first underscore)
        parts = model_stem.split('_', 1)
        if len(parts) > 1:
            timestamp = parts[1]  # e.g., "20251129_165545"
            scaler_file = model_dir / f"scaler_{timestamp}.pkl"
            if not scaler_file.exists():
                # Fallback: use latest scaler
                scaler_files = sorted(model_dir.glob("scaler_*.pkl"))
                if not scaler_files:
                    raise FileNotFoundError(f"No scaler files found in {model_path}")
                scaler_file = scaler_files[-1]
        else:
            # Fallback: use latest scaler
            scaler_files = sorted(model_dir.glob("scaler_*.pkl"))
            if not scaler_files:
                raise FileNotFoundError(f"No scaler files found in {model_path}")
            scaler_file = scaler_files[-1]
    else:
        # If model_name is specified, try to find matching scaler
        model_stem = model_file.stem
        parts = model_stem.split('_', 1)
        if len(parts) > 1:
            timestamp = parts[1]
            scaler_file = model_dir / f"scaler_{timestamp}.pkl"
            if not scaler_file.exists():
                raise FileNotFoundError(f"No matching scaler file found for {model_name}")
        else:
            # Fallback: use latest scaler
            scaler_files = sorted(model_dir.glob("scaler_*.pkl"))
            if not scaler_files:
                raise FileNotFoundError(f"No scaler files found in {model_path}")
            scaler_file = scaler_files[-1]
    
    # Load model and scaler
    model = joblib.load(model_file)
    scaler = joblib.load(scaler_file)
    
    # Get feature names (if model has them)
    if hasattr(model, 'feature_names_in_'):
        feature_names = list(model.feature_names_in_)
    else:
        # Try to get feature names from processed data
        feature_names = None
        try:
            # Get project root
            project_root = Path(__file__).parent.parent
            processed_dir = project_root / "data" / "processed"
            
            # Find latest processed data file
            processed_files = sorted(processed_dir.glob("bitcoin_processed_*.csv"))
            if processed_files:
                # Load the latest processed data to get feature columns
                df_sample = pd.read_csv(processed_files[-1], nrows=1)
                
                # Get feature columns (exclude non-feature columns)
                drop_cols = [
                    'date', 'time', 'datetime', 'extraction_time',
                    'target_price_change', 'target_direction',
                    'priceUsd'  # Don't use current price as feature
                ]
                feature_cols = [col for col in df_sample.columns if col not in drop_cols]
                
                # Verify feature count matches scaler
                if len(feature_cols) == scaler.n_features_in_:
                    feature_names = feature_cols
                    print(f"✅ Loaded {len(feature_names)} feature names from processed data")
                else:
                    print(f"⚠️  Feature count mismatch: data has {len(feature_cols)}, scaler expects {scaler.n_features_in_}")
        except Exception as e:
            print(f"⚠️  Could not load feature names from processed data: {str(e)}")
            print(f"   Scaler expects {scaler.n_features_in_} features")
    
    # Load or create metadata
    model_metadata = {
        'model_file': model_file.name,
        'scaler_file': scaler_file.name,
        'loaded_at': datetime.now().isoformat(),
        'model_type': type(model).__name__
    }
    
    # Set Prometheus metric
    model_version_info.labels(
        model_name=model_metadata['model_type'],
        version='1.0.0'
    ).set(1)
    
    print(f"✅ Model loaded: {model_file.name}")
    print(f"✅ Scaler loaded: {scaler_file.name}")
    if feature_names:
        print(f"✅ Features: {len(feature_names)}")
    
    return model, scaler


def compute_missing_features(features_dict: Dict[str, float], timestamp: Optional[str] = None) -> Dict[str, float]:
    """
    Compute missing features from provided data when possible.
    Returns a dictionary with all required features.
    """
    if not feature_names:
        return features_dict
    
    # Start with provided features
    computed_features = features_dict.copy()
    
    # Compute time features from timestamp if provided
    if timestamp:
        try:
            dt = pd.to_datetime(timestamp)
            computed_features['hour'] = dt.hour
            computed_features['day_of_week'] = dt.dayofweek
            computed_features['day_of_month'] = dt.day
            computed_features['month'] = dt.month
            
            # Cyclical encoding for hour
            if 'hour_sin' not in computed_features:
                computed_features['hour_sin'] = np.sin(2 * np.pi * dt.hour / 24)
            if 'hour_cos' not in computed_features:
                computed_features['hour_cos'] = np.cos(2 * np.pi * dt.hour / 24)
            
            # Cyclical encoding for day of week
            if 'dow_sin' not in computed_features:
                computed_features['dow_sin'] = np.sin(2 * np.pi * dt.dayofweek / 7)
            if 'dow_cos' not in computed_features:
                computed_features['dow_cos'] = np.cos(2 * np.pi * dt.dayofweek / 7)
        except Exception:
            pass  # If timestamp parsing fails, skip time feature computation
    
    # Compute rolling range from min/max if available
    for window in [3, 6, 12, 24]:
        min_key = f'priceUsd_rolling_min_{window}'
        max_key = f'priceUsd_rolling_max_{window}'
        range_key = f'priceUsd_rolling_range_{window}'
        mean_key = f'priceUsd_rolling_mean_{window}'
        std_key = f'priceUsd_rolling_std_{window}'
        
        # Compute range from min/max if both are available
        if range_key not in computed_features:
            if min_key in computed_features and max_key in computed_features:
                computed_features[range_key] = computed_features[max_key] - computed_features[min_key]
        
        # If min/max/range are missing, try to estimate from mean and std
        if min_key not in computed_features and mean_key in computed_features:
            # Estimate min as mean - 2*std, max as mean + 2*std
            if std_key in computed_features:
                computed_features[min_key] = computed_features[mean_key] - 2 * computed_features[std_key]
                computed_features[max_key] = computed_features[mean_key] + 2 * computed_features[std_key]
                if range_key not in computed_features:
                    computed_features[range_key] = 4 * computed_features[std_key]
            else:
                # If std not available, use a small percentage of mean as approximation
                computed_features[min_key] = computed_features[mean_key] * 0.98
                computed_features[max_key] = computed_features[mean_key] * 1.02
                if range_key not in computed_features:
                    computed_features[range_key] = computed_features[mean_key] * 0.04
    
    # Set default for circulatingSupply if missing (use a reasonable default)
    if 'circulatingSupply' not in computed_features:
        # Bitcoin circulating supply is around 19-20 million, use 19.5M as default
        computed_features['circulatingSupply'] = 19500000.0
    
    return computed_features


def detect_drift(features: Dict[str, float]) -> tuple[bool, float]:
    """
    Detect data drift by checking if features are within expected ranges
    Returns: (drift_detected: bool, drift_ratio: float)
    """
    if not feature_stats:
        # No baseline stats, can't detect drift
        return False, 0.0
    
    out_of_range = 0
    total_features = len(features)
    
    for feature_name, value in features.items():
        if feature_name in feature_stats:
            min_val = feature_stats[feature_name].get('min', -np.inf)
            max_val = feature_stats[feature_name].get('max', np.inf)
            
            # Check if value is outside 3 standard deviations
            mean = feature_stats[feature_name].get('mean', 0)
            std = feature_stats[feature_name].get('std', 1)
            
            if value < (mean - 3*std) or value > (mean + 3*std):
                out_of_range += 1
    
    drift_ratio = out_of_range / total_features if total_features > 0 else 0.0
    drift_detected = drift_ratio > 0.1  # 10% threshold
    
    # Update Prometheus metric
    data_drift_gauge.set(drift_ratio)
    
    return drift_detected, drift_ratio


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    try:
        load_model()
    except Exception as e:
        print(f"⚠️  Warning: Could not load model on startup: {str(e)}")
        print("   Model will need to be loaded manually via /load endpoint")


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Crypto Price Prediction API",
        "version": "1.0.0",
        "status": "online",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model is not None else "model_not_loaded",
        model_loaded=model is not None,
        model_version=model_metadata.get('model_type', 'unknown'),
        timestamp=datetime.now().isoformat()
    )


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(content=generate_latest(), media_type="text/plain")


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def get_model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return ModelInfoResponse(
        model_type=model_metadata.get('model_type', 'unknown'),
        version='1.0.0',
        features_count=len(feature_names) if feature_names else scaler.n_features_in_ if scaler else 0,
        loaded_at=model_metadata.get('loaded_at', 'unknown'),
        metadata=model_metadata
    )


@app.get("/model/features", tags=["Model"])
async def get_expected_features():
    """Get list of expected features for predictions"""
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    expected_count = scaler.n_features_in_
    
    if feature_names:
        return {
            "feature_count": len(feature_names),
            "expected_count": expected_count,
            "features": feature_names,
            "message": "All features listed below are required for predictions"
        }
    else:
        return {
            "feature_count": None,
            "expected_count": expected_count,
            "features": None,
            "message": f"Feature names not available. Please provide exactly {expected_count} features in the correct order."
        }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    """
    Make a prediction using the loaded model
    """
    if model is None or scaler is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please load model first via /load endpoint"
        )
    
    try:
        # Start timing
        with prediction_histogram.time():
            
            # Extract features
            features_dict = request.features
            
            # Compute missing features when possible
            features_dict = compute_missing_features(features_dict, request.timestamp)
            
            # Detect drift
            drift_detected, drift_ratio = detect_drift(features_dict)
            
            # Convert to DataFrame to maintain feature order
            if feature_names:
                # Check for missing features after computation
                missing_features = set(feature_names) - set(features_dict.keys())
                if missing_features:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Missing features that cannot be computed: {missing_features}. "
                               f"Required features: {feature_names}. "
                               f"You can provide a 'timestamp' field (ISO format) to compute time features automatically."
                    )
                
                # Check for extra features
                extra_features = set(features_dict.keys()) - set(feature_names)
                if extra_features:
                    print(f"⚠️  Warning: Extra features provided (will be ignored): {extra_features}")
                
                # Order features correctly
                feature_values = [features_dict[fname] for fname in feature_names]
            else:
                # Feature names not available - check feature count
                provided_count = len(features_dict)
                expected_count = scaler.n_features_in_
                
                if provided_count != expected_count:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Feature count mismatch: provided {provided_count} features, but model expects {expected_count} features. "
                               f"Please provide exactly {expected_count} features or ensure feature names are loaded from processed data."
                    )
                
                # Use provided order
                feature_values = list(features_dict.values())
            
            # Convert to numpy array
            X = np.array(feature_values).reshape(1, -1)
            
            # Validate feature count before scaling
            if X.shape[1] != scaler.n_features_in_:
                raise HTTPException(
                    status_code=400,
                    detail=f"Feature count mismatch: got {X.shape[1]} features, but scaler expects {scaler.n_features_in_} features"
                )
            
            # Scale features
            X_scaled = scaler.transform(X)
            
            # Make prediction
            prediction = float(model.predict(X_scaled)[0])
            
            # Increment counter
            prediction_counter.inc()
            
            # Determine confidence based on drift
            confidence = "low" if drift_detected else "high"
            
            return PredictionResponse(
                prediction=prediction,
                model_version=model_metadata.get('model_type', 'unknown'),
                timestamp=datetime.now().isoformat(),
                confidence=confidence,
                drift_detected=drift_detected
            )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/load", tags=["Model"])
async def load_model_endpoint(model_path: str = "models", model_name: str = None):
    """
    Load or reload the model
    """
    try:
        load_model(model_path, model_name)
        return {
            "status": "success",
            "message": "Model loaded successfully",
            "model_info": model_metadata
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


# Setup Prometheus instrumentation
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app, endpoint="/prometheus_metrics")


if __name__ == "__main__":
    import uvicorn
    
    # Load model
    try:
        load_model()
    except Exception as e:
        print(f"⚠️  Warning: {str(e)}")
    
    # Run server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )