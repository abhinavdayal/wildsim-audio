#!/usr/bin/env python3
"""
Setup a simple test dataset for Scene Creator
Uses scipy test audio files to create a working dataset
"""

import os
import sys
import shutil
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from dataset_manager import DatasetManager
from models import DatasetConfig

def setup_test_dataset():
    """Create a test dataset using scipy audio files"""
    print("🔧 Setting up test dataset for Scene Creator...")
    
    # Create directories
    os.makedirs("test_datasets", exist_ok=True)
    os.makedirs("test_datasets/forest_sounds", exist_ok=True)
    os.makedirs("test_datasets/wildlife_sounds", exist_ok=True)
    
    # Find scipy test audio files
    scipy_files = [
        ".venv/lib/python3.12/site-packages/scipy/io/tests/data/test-44100Hz-2ch-32bit-float-le.wav",
        ".venv/lib/python3.12/site-packages/scipy/io/tests/data/test-44100Hz-le-1ch-4bytes.wav",
        ".venv/lib/python3.12/site-packages/scipy/io/tests/data/test-8000Hz-le-3ch-5S-24bit.wav"
    ]
    
    # Copy files to our test dataset directories
    copied_files = []
    
    for i, scipy_file in enumerate(scipy_files):
        if os.path.exists(scipy_file):
            # Alternate between forest and wildlife directories
            if i % 2 == 0:
                dest_dir = "test_datasets/forest_sounds"
                new_name = f"forest_ambient_{i+1}.wav"
            else:
                dest_dir = "test_datasets/wildlife_sounds" 
                new_name = f"animal_sound_{i+1}.wav"
            
            dest_path = os.path.join(dest_dir, new_name)
            shutil.copy2(scipy_file, dest_path)
            copied_files.append(dest_path)
            print(f"✅ Copied: {new_name}")
    
    print(f"📁 Created {len(copied_files)} test audio files")
    
    # Now add these to the dataset manager
    dataset_manager = DatasetManager()
    
    # Add forest sounds dataset
    forest_path = os.path.abspath("test_datasets/forest_sounds")
    print(f"📊 Adding forest dataset: {forest_path}")
    forest_config = DatasetConfig(
        dataset_name="forest",
        source_type="local",
        path_or_id=forest_path,
        audio_extensions=[".wav", ".mp3", ".flac"]
    )
    dataset_manager.setup_dataset(forest_config)
    
    # Add wildlife sounds dataset  
    wildlife_path = os.path.abspath("test_datasets/wildlife_sounds")
    print(f"📊 Adding wildlife dataset: {wildlife_path}")
    wildlife_config = DatasetConfig(
        dataset_name="wildlife", 
        source_type="local",
        path_or_id=wildlife_path,
        audio_extensions=[".wav", ".mp3", ".flac"]
    )
    dataset_manager.setup_dataset(wildlife_config)
    
    # Verify the datasets were added
    catalog = dataset_manager.catalog.catalog
    datasets = list(catalog.get("datasets", {}).keys())
    files_count = len(catalog.get("files", {}))
    
    print(f"✅ Dataset setup complete!")
    print(f"   - Datasets available: {datasets}")
    print(f"   - Total audio files: {files_count}")
    
    return datasets, files_count

if __name__ == "__main__":
    try:
        datasets, files = setup_test_dataset()
        print("\n🎉 Test dataset setup successful!")
        print("🚀 You can now use Scene Creator with these datasets:")
        for dataset in datasets:
            print(f"   - {dataset}")
    except Exception as e:
        print(f"💥 Setup failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)