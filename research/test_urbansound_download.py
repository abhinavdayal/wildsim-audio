#!/usr/bin/env python3
"""
Manual test of the urbansound8k dataset download
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from dataset_manager import DatasetManager
from models import DatasetConfig

def test_urbansound8k_download():
    """Test downloading the urbansound8k dataset"""
    
    print("🎵 TESTING URBANSOUND8K DOWNLOAD")
    print("=" * 50)
    
    # Create dataset config
    config = DatasetConfig(
        dataset_name="urbansound8k",
        source_type="kaggle",
        path_or_id="chrisfilo/urbansound8k",
        audio_extensions=[".wav", ".mp3", ".flac"],
        cache_dir="./audio_cache/urbansound8k"
    )
    
    print(f"Dataset: {config.dataset_name}")
    print(f"Source: {config.path_or_id}")
    print(f"Cache: {config.cache_dir}")
    
    # Initialize dataset manager
    manager = DatasetManager()
    
    # Check if already exists
    cache_path = Path(config.cache_dir)
    if cache_path.exists():
        existing_files = list(cache_path.rglob("*"))
        print(f"\n📁 Existing cache: {len(existing_files)} items")
        
        if existing_files:
            audio_files = [f for f in existing_files if f.suffix.lower() in ['.wav', '.mp3', '.flac', '.ogg']]
            print(f"🎵 Audio files found: {len(audio_files)}")
            
            if audio_files:
                print("✅ Dataset already downloaded!")
                for af in audio_files[:5]:
                    print(f"   🎵 {af.name}")
                return True
    
    # Try to setup/download
    print(f"\n📥 Starting download...")
    success = manager.setup_dataset(config)
    
    if success:
        print("✅ Download completed successfully!")
        
        # Verify files
        final_files = list(cache_path.rglob("*"))
        audio_files = [f for f in final_files if f.suffix.lower() in ['.wav', '.mp3', '.flac', '.ogg']]
        
        print(f"📊 Final summary:")
        print(f"   Total files: {len(final_files)}")
        print(f"   Audio files: {len(audio_files)}")
        
        if audio_files:
            print(f"   Sample audio files:")
            for af in audio_files[:5]:
                size_mb = af.stat().st_size / 1024 / 1024
                print(f"     🎵 {af.name} ({size_mb:.1f}MB)")
    else:
        print("❌ Download failed!")
        
        # Check what's in the cache
        if cache_path.exists():
            all_items = list(cache_path.rglob("*"))
            print(f"Cache contents: {len(all_items)} items")
            for item in all_items[:10]:
                if item.is_file():
                    size_mb = item.stat().st_size / 1024 / 1024
                    print(f"  📄 {item.name} ({size_mb:.1f}MB)")
                else:
                    print(f"  📁 {item.name}/")
    
    return success

if __name__ == "__main__":
    test_urbansound8k_download()
