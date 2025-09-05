#!/usr/bin/env python3
"""
Test competitions and general audio datasets
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from auth_helpers import test_kaggle_dataset_access

def find_working_audio_datasets():
    """Test audio datasets and competitions"""
    
    # Mix of datasets and competitions
    audio_sources = [
        # Competitions (use competition type)
        ("birdclef-2023", "Competition"),
        ("birdclef-2022", "Competition"), 
        ("freesound-audio-tagging-2019", "Competition"),
        ("freesound-audio-tagging", "Competition"),
        
        # Regular datasets
        ("mmoreaux/audio-cats-and-dogs", "Dataset"),
        ("chrisfilo/urbansound8k", "Dataset"),
        ("karoldvl/esc50-dataset", "Dataset"),
        ("uwm-gi/environmental-sound-classification-50", "Dataset"),
        ("pavansanagapati/ad-winequality", "Dataset"),
    ]
    
    print("🎵 SEARCHING FOR WORKING AUDIO DATASETS & COMPETITIONS")
    print("=" * 65)
    
    working_sources = []
    
    for source_id, source_type in audio_sources:
        print(f"\nTesting {source_type}: {source_id}")
        result = test_kaggle_dataset_access(source_id, source_type)
        
        if result['accessible']:
            file_count = result['file_count']
            if file_count > 0:
                print(f"  ✅ WORKING! ({file_count} files)")
                if result['sample_files']:
                    print(f"     Sample files: {result['sample_files']}")
                working_sources.append({
                    'id': source_id, 
                    'type': source_type,
                    'files': file_count,
                    'samples': result['sample_files']
                })
            else:
                print(f"  ⚠️  Accessible but empty (0 files)")
        else:
            print(f"  ❌ {result['error']}")
    
    print(f"\n🎉 WORKING AUDIO SOURCES:")
    print("=" * 40)
    for source in working_sources:
        print(f"📦 {source['type']}: {source['id']}")
        print(f"   Files: {source['files']}")
        if source['samples']:
            print(f"   Sample files: {source['samples'][:3]}")
        print()
    
    return working_sources

if __name__ == "__main__":
    working = find_working_audio_datasets()
    
    if working:
        print("🔧 RECOMMENDATIONS:")
        print("For your wildlife simulation, try:")
        for source in working:
            if source['files'] > 5:  # Only recommend datasets with multiple files
                print(f"  • {source['id']} ({source['files']} files)")
    else:
        print("❌ No working audio datasets found")
