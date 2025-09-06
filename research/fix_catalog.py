#!/usr/bin/env python3
"""
Fix the corrupted dataset catalog by rebuilding it from scratch
"""

import os
import sys
import json
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from dataset_manager import DatasetManager
from models import DatasetConfig

def rebuild_catalog():
    """Rebuild the catalog from scratch"""
    print("🔧 Rebuilding corrupted dataset catalog...")
    
    # Backup the old catalog
    catalog_path = "audio_cache/audio_catalog.json"
    if os.path.exists(catalog_path):
        backup_path = "audio_cache/audio_catalog_backup.json"
        os.rename(catalog_path, backup_path)
        print(f"📦 Backed up corrupted catalog to {backup_path}")
    
    # Create fresh dataset manager
    dataset_manager = DatasetManager()
    
    # Check what datasets actually exist in cache
    cache_dir = Path("audio_cache")
    existing_dirs = [d for d in cache_dir.iterdir() if d.is_dir()]
    
    print("📁 Found existing directories:")
    for d in existing_dirs:
        audio_count = len(list(d.rglob("*.wav")) + list(d.rglob("*.mp3")) + list(d.rglob("*.flac")))
        print(f"   {d.name}: {audio_count} audio files")
    
    # Rebuild datasets based on actual directory contents
    datasets_to_rebuild = []
    
    # Check each directory
    for dataset_dir in existing_dirs:
        if dataset_dir.name in ['forest', 'wildlife']:
            # These are our local test datasets - skip for now
            continue
            
        audio_files = (list(dataset_dir.rglob("*.wav")) + 
                      list(dataset_dir.rglob("*.mp3")) + 
                      list(dataset_dir.rglob("*.flac")))
        
        if audio_files:
            datasets_to_rebuild.append((dataset_dir.name, str(dataset_dir.absolute()), len(audio_files)))
    
    print(f"🔄 Will rebuild {len(datasets_to_rebuild)} datasets:")
    
    # Rebuild each dataset
    for dataset_name, dataset_path, file_count in datasets_to_rebuild:
        print(f"📊 Rebuilding {dataset_name} ({file_count} files)...")
        
        try:
            config = DatasetConfig(
                dataset_name=dataset_name,
                source_type="local", 
                path_or_id=dataset_path,
                audio_extensions=[".wav", ".mp3", ".flac"]
            )
            
            success = dataset_manager.setup_dataset(config)
            if success:
                print(f"✅ Successfully rebuilt {dataset_name}")
            else:
                print(f"❌ Failed to rebuild {dataset_name}")
                
        except Exception as e:
            print(f"❌ Error rebuilding {dataset_name}: {e}")
    
    # Add our test datasets back
    print("🧪 Re-adding test datasets...")
    
    # Forest dataset
    if os.path.exists("test_datasets/forest_sounds"):
        forest_config = DatasetConfig(
            dataset_name="forest",
            source_type="local",
            path_or_id=os.path.abspath("test_datasets/forest_sounds"),
            audio_extensions=[".wav", ".mp3", ".flac"]
        )
        dataset_manager.setup_dataset(forest_config)
        print("✅ Re-added forest dataset")
    
    # Wildlife dataset  
    if os.path.exists("test_datasets/wildlife_sounds"):
        wildlife_config = DatasetConfig(
            dataset_name="wildlife",
            source_type="local", 
            path_or_id=os.path.abspath("test_datasets/wildlife_sounds"),
            audio_extensions=[".wav", ".mp3", ".flac"]
        )
        dataset_manager.setup_dataset(wildlife_config)
        print("✅ Re-added wildlife dataset")
    
    # Verify the rebuild
    catalog = dataset_manager.catalog.catalog
    datasets = catalog.get("datasets", {})
    files = catalog.get("files", {})
    
    print("\n📊 Rebuilt catalog summary:")
    for dataset_name, dataset_info in datasets.items():
        actual_files = [f for f in files.values() if f.get("dataset") == dataset_name]
        print(f"   {dataset_name}: {len(actual_files)} files")
    
    print(f"\n✅ Catalog rebuild complete! Total files: {len(files)}")
    return datasets, len(files)

if __name__ == "__main__":
    try:
        datasets, file_count = rebuild_catalog()
        print(f"\n🎉 Catalog rebuild successful!")
        print(f"📚 Datasets: {list(datasets.keys())}")
        print(f"📁 Total files: {file_count}")
    except Exception as e:
        print(f"💥 Rebuild failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)