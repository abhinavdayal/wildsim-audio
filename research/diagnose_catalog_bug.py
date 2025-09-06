#!/usr/bin/env python3
"""
Diagnose the dataset catalog bug by tracing through the dataset setup process
"""

import os
import sys
import json
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from dataset_manager import DatasetManager
from models import DatasetConfig

def trace_dataset_setup():
    """Trace through dataset setup to find the bug"""
    print("🔍 Diagnosing dataset catalog bug...")
    
    # Check what's actually in the cache directories
    cache_dir = Path("audio_cache")
    
    print("\n📁 Physical directory structure:")
    for dataset_dir in cache_dir.iterdir():
        if dataset_dir.is_dir():
            audio_files = (list(dataset_dir.rglob("*.wav")) + 
                          list(dataset_dir.rglob("*.mp3")) + 
                          list(dataset_dir.rglob("*.flac")))
            print(f"   {dataset_dir.name}: {len(audio_files)} audio files")
            
            # Show some example paths
            for f in audio_files[:3]:
                print(f"      {f.relative_to(cache_dir)}")
    
    # Load the current catalog to see what's wrong
    print("\n📚 Current catalog mapping:")
    if os.path.exists("audio_cache/audio_catalog.json"):
        with open("audio_cache/audio_catalog.json", 'r') as f:
            catalog = json.load(f)
        
        datasets = catalog.get("datasets", {})
        files = catalog.get("files", {})
        
        for dataset_name, dataset_info in datasets.items():
            dataset_files = [f for f in files.values() if f.get("dataset") == dataset_name]
            config = dataset_info.get("config", {})
            path_or_id = config.get("path_or_id", "unknown")
            
            print(f"   {dataset_name}:")
            print(f"      Config path: {path_or_id}")
            print(f"      Mapped files: {len(dataset_files)}")
            
            # Show where files actually point
            actual_paths = set()
            for file_info in dataset_files[:5]:  # First 5 files
                actual_path = file_info.get("absolute_path", "")
                if actual_path:
                    # Extract the directory this file is actually in
                    actual_dir = Path(actual_path).parent
                    actual_paths.add(str(actual_dir))
            
            if actual_paths:
                print(f"      Files actually from: {actual_paths}")
            
            print()
    
    # Test the scan_dataset method directly
    print("🧪 Testing scan_dataset method:")
    
    # Create a fresh DatasetManager 
    dataset_manager = DatasetManager()
    
    # Test scanning wild_animals directory directly
    wild_animals_path = cache_dir / "wild_animals"
    if wild_animals_path.exists():
        print(f"\n🔬 Testing scan of {wild_animals_path}")
        
        config = DatasetConfig(
            dataset_name="test_wild_animals",
            source_type="local",
            path_or_id=str(wild_animals_path.absolute()),
            audio_extensions=[".wav", ".mp3", ".flac"]
        )
        
        # Call scan_dataset directly to see what happens
        result = dataset_manager.catalog.scan_dataset(config)
        print(f"   Scanned {len(result)} files")
        
        # Check what dataset name they got assigned
        for rel_path, file_info in list(result.items())[:3]:
            print(f"   {rel_path}: dataset='{file_info.get('dataset')}', path='{file_info.get('absolute_path')}'")

def trace_hash_method():
    """Test if there's an issue with the hash method"""
    print("\n🔐 Testing hash calculation:")
    
    from dataset_manager import AudioCatalog
    
    catalog = AudioCatalog()
    
    # Test hash calculation on some files
    cache_dir = Path("audio_cache")
    test_files = []
    
    for dataset_dir in cache_dir.iterdir():
        if dataset_dir.is_dir():
            audio_files = list(dataset_dir.glob("*.wav"))[:2]  # Get 2 files from each
            test_files.extend(audio_files)
    
    print(f"   Testing {len(test_files)} files for hash collisions:")
    
    hashes = {}
    collisions = []
    
    for file_path in test_files:
        try:
            file_hash = catalog._calculate_file_hash(file_path)
            if file_hash in hashes:
                collisions.append((file_hash, hashes[file_hash], file_path))
                print(f"   ❌ COLLISION! Hash {file_hash}:")
                print(f"      File 1: {hashes[file_hash]}")
                print(f"      File 2: {file_path}")
            else:
                hashes[file_hash] = file_path
                print(f"   ✅ {file_path.name}: {file_hash}")
        except Exception as e:
            print(f"   ❌ Error hashing {file_path}: {e}")
    
    print(f"\n   Found {len(collisions)} hash collisions")
    return len(collisions) > 0

if __name__ == "__main__":
    trace_dataset_setup()
    has_collisions = trace_hash_method()
    
    if has_collisions:
        print("\n💥 Found hash collisions! This explains the dataset mapping bug.")
    else:
        print("\n🤔 No hash collisions found. Bug might be elsewhere in the catalog logic.")