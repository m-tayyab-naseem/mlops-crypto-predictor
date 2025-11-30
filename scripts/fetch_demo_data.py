"""
Fetch demo data for CI/CD testing
"""
import sys
sys.path.append('.')
from scripts.create_sample_data import create_sample_data

if __name__ == "__main__":
    create_sample_data('data/processed', 100)