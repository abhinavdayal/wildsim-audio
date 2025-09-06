#!/usr/bin/env python3
"""
Script to catalog the wild animals dataset with the unified labeling system
"""

import sys
sys.path.append('.')

from models import DatasetConfig
from dataset_manager import DatasetManager
from pathlib import Path

def catalog_wild_animals_dataset():
    """Catalog the wild animals dataset"""
    
    print("🚀 Cataloging Wild Animals Dataset")
    print("=" * 40)
    
    # Create dataset manager
    manager = DatasetManager("./audio_cache")
    
    # Define wild animals dataset configuration
    wild_animals_config = DatasetConfig(
        dataset_name="wild_animals",
        source_type="local",
        path_or_id="./audio_cache/wild_animals",
        audio_extensions=[".wav"],
        cache_dir="./audio_cache"
    )
    
    print(f"📁 Dataset path: {wild_animals_config.path_or_id}")
    
    # Check if dataset exists
    dataset_path = Path(wild_animals_config.path_or_id)
    if not dataset_path.exists():
        print(f"❌ Dataset path not found: {dataset_path}")
        return False
    
    # Setup and catalog the dataset
    print("📊 Setting up wild animals dataset...")
    success = manager.setup_dataset(wild_animals_config)
    
    if success:
        print("✅ Wild animals dataset cataloged successfully!")
        
        # Get updated summary
        summary = manager.get_dataset_summary()
        print(f"\n📋 Updated Dataset Summary:")
        print(f"   Total datasets: {summary['total_datasets']}")
        print(f"   Total files: {summary['total_files']}")
        
        # Show source type distribution
        sorted_types = sorted(summary['source_type_counts'].items(), key=lambda x: x[1], reverse=True)
        print(f"\n📊 Source Type Distribution:")
        for source_type, count in sorted_types:
            print(f"   {source_type}: {count} files")
        
        # Test category access for wild animals
        print(f"\n🔍 Testing Wild Animals Categories:")
        
        wild_animal_categories = ['elephant', 'lion', 'dog', 'cat', 'bear', 'horse']
        for category in wild_animal_categories:
            files = manager.get_files_by_category(category, limit=3)
            if files:
                print(f"   {category}: {len(files)} files found")
                for file_info in files[:2]:
                    filename = Path(file_info['absolute_path']).name
                    print(f"      - {filename}")
            else:
                print(f"   {category}: No files found")
        
        return True
    else:
        print("❌ Failed to catalog wild animals dataset")
        return False

if __name__ == "__main__":
    success = catalog_wild_animals_dataset()
    
    if success:
        print("\n🎯 Wild animals dataset successfully integrated!")
        print("✅ Unified labeling system now covers:")
        print("   - Forest sounds (27 categories)")
        print("   - Wild animals (13 categories)")
        print("   - Original categories (9 categories)")
        print("   - Total: 44+ sound categories")
    else:
        print("\n❌ Failed to integrate wild animals dataset")
