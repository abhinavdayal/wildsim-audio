#!/usr/bin/env python3
"""
Test script to debug Kaggle dataset loading
"""

import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

from auth_helpers import test_kaggle_dataset_access, check_kaggle_auth

def test_specific_dataset():
    """Test the specific dataset you're trying to use"""
    
    print("=" * 60)
    print("KAGGLE DATASET LOADING TEST")
    print("=" * 60)
    
    # First, check authentication
    print("\n1. Checking Kaggle Authentication...")
    auth_result = check_kaggle_auth()
    
    if auth_result['authenticated']:
        print(f"✅ Authentication successful!")
        print(f"   Username: {auth_result['username']}")
    else:
        print(f"❌ Authentication failed!")
        print(f"   Error: {auth_result['error']}")
        return
    
    # Test the specific dataset
    dataset_id = "imoore/xenocanto-bird-recordings-dataset"
    print(f"\n2. Testing dataset access: {dataset_id}")
    
    test_result = test_kaggle_dataset_access(dataset_id, "Dataset")
    
    if test_result['accessible']:
        print(f"✅ Dataset is accessible!")
        print(f"   File count: {test_result['file_count']}")
        print(f"   Sample files: {test_result['sample_files']}")
        print(f"   URL: {test_result['dataset_url']}")
    else:
        print(f"❌ Dataset access failed!")
        print(f"   Error: {test_result['error']}")
        print(f"   Suggestion: {test_result['suggestion']}")
    
    # Test some alternative popular audio datasets
    print(f"\n3. Testing other known working audio datasets...")
    
    test_datasets = [
        "karoldvl/esc50-dataset",
        "mmoreaux/audio-cats-and-dogs",
        "chrisfilo/urbansound8k"
    ]
    
    for dataset in test_datasets:
        print(f"\nTesting: {dataset}")
        result = test_kaggle_dataset_access(dataset, "Dataset")
        if result['accessible']:
            print(f"  ✅ Accessible ({result['file_count']} files)")
        else:
            print(f"  ❌ Not accessible: {result['error']}")

if __name__ == "__main__":
    test_specific_dataset()
