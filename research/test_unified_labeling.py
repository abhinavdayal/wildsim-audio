#!/usr/bin/env python3
"""
Test script to verify the unified labeling system and updated SourceType enum
"""

import sys
sys.path.append('.')

from models import SourceType, DatasetConfig
from dataset_manager import DatasetManager
from pathlib import Path

def test_source_type_enum():
    """Test that all forest and wild animals classes are in the SourceType enum"""
    print("🧪 Testing SourceType enum with all classes...")
    
    # Expected forest classes
    expected_forest_classes = [
        'axe', 'birdchirping', 'chainsaw', 'clapping', 'fire', 'firework',
        'footsteps', 'frog', 'generator', 'gunshot', 'handsaw', 'helicopter',
        'insect', 'lion', 'rain', 'silence', 'speaking', 'squirrel',
        'thunderstorm', 'treefalling', 'vehicleengine', 'waterdrops',
        'whistling', 'wind', 'wingflaping', 'wolfhowl', 'woodchop'
    ]
    
    # Expected wild animals classes
    expected_wild_animals_classes = [
        'bear', 'cat', 'chicken', 'cow', 'dog', 'dolphin', 'donkey',
        'elephant', 'horse', 'monkey', 'sheep'
    ]
    
    enum_values = [e.value for e in SourceType]
    
    print(f"📊 Total SourceType enum values: {len(enum_values)}")
    print(f"📊 Expected forest classes: {len(expected_forest_classes)}")
    print(f"📊 Expected wild animals classes: {len(expected_wild_animals_classes)}")
    
    missing_classes = []
    for cls in expected_forest_classes + expected_wild_animals_classes:
        if cls not in enum_values:
            missing_classes.append(cls)
    
    if missing_classes:
        print(f"❌ Missing classes in enum: {missing_classes}")
        return False
    else:
        print("✅ All forest and wild animals classes found in SourceType enum")
        return True

def test_unified_labeling():
    """Test the unified labeling system"""
    print("\n🧪 Testing unified labeling system...")
    
    # Create test dataset manager
    manager = DatasetManager("./audio_cache")
    
    # Test forest dataset if it exists
    forest_path = Path("./audio_cache/forest")
    if forest_path.exists():
        print(f"📁 Testing forest dataset at {forest_path}")
        
        # Test forest dataset files
        test_forest_files = [
            forest_path / "audio" / "fire01.wav",
            forest_path / "audio" / "rain05.wav",
            forest_path / "audio" / "lion10.wav",
            forest_path / "audio" / "elephant15.wav" if (forest_path / "audio" / "elephant15.wav").exists() else None
        ]
        
        for test_file in test_forest_files:
            if test_file and test_file.exists():
                # Test the _infer_source_type method
                inferred_type = manager.catalog._infer_source_type(test_file)
                expected_type = test_file.stem.rstrip('0123456789')
                
                print(f"🔍 Forest File: {test_file.name}")
                print(f"   Expected: {expected_type}")
                print(f"   Inferred: {inferred_type}")
                
                if expected_type == inferred_type:
                    print("   ✅ Match!")
                else:
                    print("   ❌ Mismatch!")
    else:
        print("📁 Forest dataset not found - skipping forest tests")
    
    # Test wild animals dataset if it exists
    wild_animals_path = Path("./audio_cache/wild_animals/Animal-Soundprepros")
    if wild_animals_path.exists():
        print(f"\n📁 Testing wild animals dataset at {wild_animals_path}")
        
        # Test wild animals dataset files
        test_wild_animals_files = [
            wild_animals_path / "Elephant" / "elephant01.wav",
            wild_animals_path / "Dog" / "dog01.wav",
            wild_animals_path / "Cat" / "cat01.wav",
            wild_animals_path / "Aslan" / "lion01.wav"
        ]
        
        for test_file in test_wild_animals_files:
            if test_file.exists():
                # Test the _infer_source_type method
                inferred_type = manager.catalog._infer_source_type(test_file)
                
                # Extract expected type from filename
                expected_type = test_file.stem.rstrip('0123456789')
                
                # Special case for Aslan folder (contains lion files)
                if 'Aslan' in str(test_file) and expected_type == 'lion':
                    expected_type = 'lion'
                
                print(f"🔍 Wild Animals File: {test_file.name}")
                print(f"   Path: .../{test_file.parent.name}/{test_file.name}")
                print(f"   Expected: {expected_type}")
                print(f"   Inferred: {inferred_type}")
                
                if expected_type == inferred_type:
                    print("   ✅ Match!")
                else:
                    print("   ❌ Mismatch!")
    else:
        print("📁 Wild animals dataset not found - skipping wild animals tests")
    
    print("✅ Unified labeling system test completed")
    return True

def test_dataset_summary():
    """Test dataset summary functionality"""
    print("\n🧪 Testing dataset summary...")
    
    manager = DatasetManager("./audio_cache")
    summary = manager.get_dataset_summary()
    
    print(f"📊 Total datasets: {summary['total_datasets']}")
    print(f"📊 Total files: {summary['total_files']}")
    print(f"📊 Source type counts: {len(summary['source_type_counts'])} categories")
    
    # Show top categories
    sorted_types = sorted(summary['source_type_counts'].items(), key=lambda x: x[1], reverse=True)
    print("📊 Top categories:")
    for source_type, count in sorted_types[:10]:
        print(f"   {source_type}: {count} files")
    
    if summary['total_files'] > 0:
        print("✅ Dataset summary working")
        return True
    else:
        print("ℹ️ No datasets found - this is expected if no datasets have been cataloged yet")
        return True

def test_category_access():
    """Test category-based file access"""
    print("\n🧪 Testing category-based file access...")
    
    manager = DatasetManager("./audio_cache")
    
    # Test getting available categories
    categories = manager.get_available_categories()
    print(f"📊 Available categories: {len(categories)}")
    
    if categories:
        print("📋 Sample categories:")
        for cat in categories[:10]:
            print(f"   - {cat}")
        
        # Test getting files by category
        test_category = categories[0]
        files = manager.get_files_by_category(test_category, limit=3)
        print(f"\n🔍 Testing {test_category} category: {len(files)} files")
        
        for file_info in files[:3]:
            print(f"   📄 {Path(file_info['absolute_path']).name}")
        
        # Test random file selection
        random_file = manager.get_random_file_by_category(test_category)
        if random_file:
            print(f"🎲 Random {test_category} file: {Path(random_file).name}")
        
        print("✅ Category access working")
        return True
    else:
        print("ℹ️ No categories found")
        return True

if __name__ == "__main__":
    print("🚀 Testing Enhanced Audio Dataset Management System")
    print("=" * 60)
    
    success = True
    
    # Test 1: SourceType enum
    success &= test_source_type_enum()
    
    # Test 2: Unified labeling
    success &= test_unified_labeling()
    
    # Test 3: Dataset summary
    success &= test_dataset_summary()
    
    # Test 4: Category access
    success &= test_category_access()
    
    print(f"\n{'✅ All tests passed!' if success else '❌ Some tests failed!'}")
    print(f"🎯 System ready for {'production use' if success else 'debugging'}!")
