#!/usr/bin/env python3
"""
Find better bird/wildlife audio datasets on Kaggle
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from auth_helpers import test_kaggle_dataset_access

def find_working_bird_datasets():
    """Test various bird/wildlife audio datasets"""
    
    # Known bird/wildlife audio datasets to test
    bird_datasets = [
        "rtatman/bird-song-classification",
        "vinayshanbhag/bird-species-audio-detection-dataset", 
        "gpiosenka/birds-525-species-image-classification",
        "warcoder/bird-species-audio-classification",
        "saurabhshahane/music-dataset-1950-to-2019",
        "imsparsh/birdsong-dataset-from-xeno-canto",
        "ryanholbrook/birdsong-classification",
        "tommyngx/freesound-audio-tagging-2019"
    ]
    
    print("🐦 SEARCHING FOR WORKING BIRD/WILDLIFE AUDIO DATASETS")
    print("=" * 65)
    
    working_datasets = []
    
    for dataset_id in bird_datasets:
        print(f"\nTesting: {dataset_id}")
        result = test_kaggle_dataset_access(dataset_id, "Dataset")
        
        if result['accessible']:
            file_count = result['file_count']
            if file_count > 0:
                print(f"  ✅ WORKING! ({file_count} files)")
                print(f"     Sample files: {result['sample_files']}")
                working_datasets.append({
                    'id': dataset_id, 
                    'files': file_count,
                    'samples': result['sample_files']
                })
            else:
                print(f"  ⚠️  Accessible but empty (0 files)")
        else:
            print(f"  ❌ {result['error']}")
    
    print(f"\n🎉 WORKING DATASETS SUMMARY:")
    print("=" * 40)
    for dataset in working_datasets:
        print(f"📦 {dataset['id']}")
        print(f"   Files: {dataset['files']}")
        print(f"   Samples: {dataset['samples'][:3]}")
        print()

if __name__ == "__main__":
    find_working_bird_datasets()
