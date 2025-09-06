#!/usr/bin/env python3
"""
Complete fix for the catalog - properly catalog ALL datasets including the missing large ones
"""

import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from dataset_manager import DatasetManager
from models import DatasetConfig

def complete_catalog_rebuild():
    """Rebuild catalog with ALL datasets properly included"""
    print("🔧 Complete catalog rebuild - including missing large datasets...")
    
    # Backup the current catalog
    catalog_path = "audio_cache/audio_catalog.json"
    if os.path.exists(catalog_path):
        backup_path = "audio_cache/audio_catalog_partial_backup.json"
        os.rename(catalog_path, backup_path)
        print(f"📦 Backed up partial catalog to {backup_path}")
    
    # Create fresh dataset manager
    dataset_manager = DatasetManager()
    
    # Map of actual dataset directories to proper names
    datasets_to_catalog = [
        # Large forest dataset (2025 files) - was missing!
        ("forest_large", "audio_cache/forest", "Large forest ambient sound collection"),
        
        # Birds dataset (5 files)
        ("birds_sample", "audio_cache/birds_sample", "Sample bird sounds"),
        
        # Elephants dataset (8 files) 
        ("elephants_sample", "audio_cache/elephants_sample", "Sample elephant sounds"),
        
        # Wild animals dataset (650 files)
        ("wild_animals", "audio_cache/wild_animals", "Wild animal sounds collection"),
        
        # Our test datasets (small)
        ("forest_test", "test_datasets/forest_sounds", "Test forest sounds"),
        ("wildlife_test", "test_datasets/wildlife_sounds", "Test wildlife sounds"),
    ]
    
    print(f"📊 Will catalog {len(datasets_to_catalog)} datasets:")
    
    total_files = 0
    success_count = 0
    
    # Catalog each dataset
    for dataset_name, dataset_path, description in datasets_to_catalog:
        abs_path = os.path.abspath(dataset_path)
        
        if not os.path.exists(abs_path):
            print(f"⚠️  Skipping {dataset_name} - path not found: {abs_path}")
            continue
            
        # Count files first
        audio_files = []
        for ext in [".wav", ".mp3", ".flac"]:
            audio_files.extend(list(Path(abs_path).rglob(f"*{ext}")))
        
        print(f"📁 {dataset_name}: {len(audio_files)} files at {abs_path}")
        
        if len(audio_files) == 0:
            print(f"⚠️  Skipping {dataset_name} - no audio files found")
            continue
            
        try:
            config = DatasetConfig(
                dataset_name=dataset_name,
                source_type="local", 
                path_or_id=abs_path,
                audio_extensions=[".wav", ".mp3", ".flac"]
            )
            
            success = dataset_manager.setup_dataset(config)
            if success:
                print(f"✅ Successfully cataloged {dataset_name}")
                total_files += len(audio_files)
                success_count += 1
            else:
                print(f"❌ Failed to catalog {dataset_name}")
                
        except Exception as e:
            print(f"❌ Error cataloging {dataset_name}: {e}")
    
    # Verify the complete rebuild
    catalog = dataset_manager.catalog.catalog
    datasets = catalog.get("datasets", {})
    files = catalog.get("files", {})
    
    print(f"\n📊 Complete catalog summary:")
    total_cataloged = 0
    for dataset_name, dataset_info in datasets.items():
        actual_files = [f for f in files.values() if f.get("dataset") == dataset_name]
        print(f"   {dataset_name}: {len(actual_files)} files")
        total_cataloged += len(actual_files)
    
    print(f"\n✅ Complete catalog rebuild finished!")
    print(f"📚 Datasets: {success_count}/{len(datasets_to_catalog)} successful")
    print(f"📁 Total files cataloged: {total_cataloged}")
    
    return datasets, total_cataloged

if __name__ == "__main__":
    try:
        datasets, file_count = complete_catalog_rebuild()
        print(f"\n🎉 Complete catalog rebuild successful!")
        print(f"📚 Datasets: {list(datasets.keys())}")
        print(f"📁 Total files: {file_count}")
    except Exception as e:
        print(f"💥 Rebuild failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)