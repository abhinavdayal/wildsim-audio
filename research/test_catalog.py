#!/usr/bin/env python3
"""
Test the dataset catalog scanning
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from dataset_manager import DatasetManager
from models import DatasetConfig

def test_catalog_scan():
    """Test cataloging the urbansound8k dataset"""
    
    print("🎵 TESTING DATASET CATALOGING")
    print("=" * 50)
    
    # Create dataset config
    config = DatasetConfig(
        dataset_name="urbansound8k",
        source_type="kaggle",
        path_or_id="chrisfilo/urbansound8k",
        audio_extensions=[".wav", ".mp3", ".flac"],
        cache_dir="./audio_cache/urbansound8k"
    )
    
    # Initialize dataset manager
    manager = DatasetManager()
    
    # Force a fresh catalog scan
    print("🔍 Starting fresh catalog scan...")
    
    # Clear the catalog for this dataset if it exists
    if config.dataset_name in manager.catalog.catalog["datasets"]:
        print("Removing existing catalog entry...")
        del manager.catalog.catalog["datasets"][config.dataset_name]
        
        # Remove file entries for this dataset
        files_to_remove = []
        for file_hash, file_info in manager.catalog.catalog["files"].items():
            if file_info.get("dataset") == config.dataset_name:
                files_to_remove.append(file_hash)
        
        for file_hash in files_to_remove:
            del manager.catalog.catalog["files"][file_hash]
    
    # Setup dataset (should now catalog the files)
    success = manager.setup_dataset(config)
    
    if success:
        print("✅ Setup completed successfully!")
        
        # Get summary
        summary = manager.get_dataset_summary()
        
        print(f"\n📊 DATASET SUMMARY:")
        print(f"   Total datasets: {summary['total_datasets']}")
        print(f"   Total files: {summary['total_files']}")
        
        if summary['source_type_counts']:
            print(f"\n🏷️  Source types found:")
            for source_type, count in summary['source_type_counts'].items():
                print(f"     {source_type}: {count} files")
        
        if summary['format_counts']:
            print(f"\n🎵 File formats:")
            for format_ext, count in summary['format_counts'].items():
                print(f"     {format_ext}: {count} files")
        
        # Test finding specific audio files
        print(f"\n🔎 TESTING AUDIO FILE SEARCH:")
        
        # Find some vehicle sounds (should be in urban sound dataset)
        matches = manager.catalog.find_audio_files(min_duration=1.0, max_duration=5.0)
        print(f"   Audio files (1-5 seconds): {len(matches)}")
        
        if matches:
            print(f"   Sample files:")
            for match in matches[:3]:
                duration = match.get('duration', 0)
                source_type = match.get('source_type', 'unknown')
                file_path = Path(match.get('absolute_path', ''))
                print(f"     🎵 {file_path.name} ({duration:.1f}s, {source_type})")
    
    else:
        print("❌ Setup failed!")

if __name__ == "__main__":
    test_catalog_scan()
