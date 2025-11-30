"""
Create sample data for CI/CD testing
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

def create_sample_data(output_path='data/processed', n_samples=100):
    """Create sample cryptocurrency data with features"""
    
    print(f"Creating sample data with {n_samples} samples...")
    
    # Create timestamps
    base_time = datetime.now() - timedelta(hours=n_samples)
    timestamps = [base_time + timedelta(hours=i) for i in range(n_samples)]
    
    # Create features
    np.random.seed(42)
    
    data = {
        # Time features
        'datetime': timestamps,
        'hour': [t.hour for t in timestamps],
        'day_of_week': [t.weekday() for t in timestamps],
        'hour_sin': [np.sin(2 * np.pi * t.hour / 24) for t in timestamps],
        'hour_cos': [np.cos(2 * np.pi * t.hour / 24) for t in timestamps],
        'dow_sin': [np.sin(2 * np.pi * t.weekday() / 7) for t in timestamps],
        'dow_cos': [np.cos(2 * np.pi * t.weekday() / 7) for t in timestamps],
        
        # Price features (simulated Bitcoin price around $45,000)
        'priceUsd_lag_1': 45000 + np.random.randn(n_samples) * 1000,
        'priceUsd_lag_2': 45000 + np.random.randn(n_samples) * 1000,
        'priceUsd_lag_3': 45000 + np.random.randn(n_samples) * 1000,
        'priceUsd_lag_6': 45000 + np.random.randn(n_samples) * 1000,
        'priceUsd_lag_12': 45000 + np.random.randn(n_samples) * 1000,
        'priceUsd_lag_24': 45000 + np.random.randn(n_samples) * 1000,
        
        # Rolling features
        'priceUsd_rolling_mean_3': 45000 + np.random.randn(n_samples) * 800,
        'priceUsd_rolling_mean_6': 45000 + np.random.randn(n_samples) * 800,
        'priceUsd_rolling_mean_12': 45000 + np.random.randn(n_samples) * 800,
        'priceUsd_rolling_mean_24': 45000 + np.random.randn(n_samples) * 800,
        
        'priceUsd_rolling_std_3': np.abs(np.random.randn(n_samples) * 100),
        'priceUsd_rolling_std_6': np.abs(np.random.randn(n_samples) * 100),
        'priceUsd_rolling_std_12': np.abs(np.random.randn(n_samples) * 100),
        'priceUsd_rolling_std_24': np.abs(np.random.randn(n_samples) * 100),
        
        'priceUsd_rolling_min_3': 44000 + np.random.randn(n_samples) * 500,
        'priceUsd_rolling_max_3': 46000 + np.random.randn(n_samples) * 500,
        'priceUsd_rolling_range_3': np.abs(np.random.randn(n_samples) * 500),
        
        # Price change features
        'price_pct_change_1h': np.random.randn(n_samples) * 0.01,
        'price_pct_change_3h': np.random.randn(n_samples) * 0.02,
        'price_pct_change_6h': np.random.randn(n_samples) * 0.03,
        'price_pct_change_12h': np.random.randn(n_samples) * 0.04,
        'price_pct_change_24h': np.random.randn(n_samples) * 0.05,
        
        'price_diff_1h': np.random.randn(n_samples) * 200,
        'price_diff_6h': np.random.randn(n_samples) * 500,
        
        'momentum_6h': np.random.randn(n_samples) * 2,
        'momentum_12h': np.random.randn(n_samples) * 3,
        
        'volatility_6h': np.abs(np.random.randn(n_samples) * 0.01),
        'volatility_12h': np.abs(np.random.randn(n_samples) * 0.01),
        'volatility_24h': np.abs(np.random.randn(n_samples) * 0.015),
        
        # Target variable
        'target_price_change': np.random.randn(n_samples) * 0.02,
        'target_direction': np.random.randint(0, 2, n_samples)
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to file
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'bitcoin_processed_sample_{timestamp}.csv'
    
    df.to_csv(output_file, index=False)
    
    print(f"✅ Sample data created: {output_file}")
    print(f"   Shape: {df.shape}")
    print(f"   Features: {len(df.columns)}")
    
    return str(output_file)


if __name__ == "__main__":
    import sys
    
    n_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    output_path = sys.argv[2] if len(sys.argv) > 2 else 'data/processed'
    
    create_sample_data(output_path, n_samples)