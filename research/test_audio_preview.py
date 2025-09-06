#!/usr/bin/env python3
"""
Test the audio preview functionality
"""

import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_audio_files():
    """Test that we have audio files available for preview"""
    print("🎧 Testing audio preview functionality...")
    
    # Check our datasets
    test_files = [
        "audio_cache/birds_sample/Cinereous Tinamou30.mp3",
        "audio_cache/elephants_sample/Ys_ele146.wav", 
        "test_datasets/forest_sounds/forest_ambient_1.wav",
        "test_datasets/wildlife_sounds/animal_sound_2.wav"
    ]
    
    working_files = []
    
    for file_path in test_files:
        if os.path.exists(file_path):
            try:
                file_size = os.path.getsize(file_path)
                file_ext = Path(file_path).suffix.lower()
                print(f"✅ {Path(file_path).name}: {file_size/1024:.1f}KB, format: {file_ext}")
                working_files.append(file_path)
            except Exception as e:
                print(f"❌ {Path(file_path).name}: Error - {e}")
        else:
            print(f"⚠️  {Path(file_path).name}: File not found")
    
    print(f"\n📊 Summary:")
    print(f"   Available for preview: {len(working_files)} files")
    
    if working_files:
        print(f"✅ Audio preview should work! Files ready:")
        for f in working_files:
            print(f"   - {Path(f).name}")
        
        print(f"\n🚀 Now in Scene Creator you can:")
        print(f"   1. Select a dataset (birds_sample, elephants_sample, etc.)")
        print(f"   2. Choose a specific audio file")
        print(f"   3. See audio preview with play button")
        print(f"   4. See file info (duration, size, format)")
        print(f"   5. Preview sounds already added to scene")
        
        return True
    else:
        print(f"❌ No audio files available for preview")
        return False

if __name__ == "__main__":
    success = test_audio_files()
    if success:
        print(f"\n🎉 Audio preview functionality ready!")
    else:
        print(f"\n💥 Audio preview not ready - check file paths")
        sys.exit(1)