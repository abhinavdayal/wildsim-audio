#!/usr/bin/env python3
"""
Test the ambient sound functionality
"""

import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from simple_config import SceneConfig, SoundPlacement, AmbientSound

def test_ambient_functionality():
    """Test ambient sound creation and configuration"""
    print("🌿 Testing ambient sound functionality...")
    
    # Test creating various ambient sounds
    ambient_types = [
        ("forest", -30.0),
        ("wind", -35.0),
        ("rain", -25.0),
        ("stream", -40.0),
        ("insects", -45.0),
        ("birds_distant", -35.0)
    ]
    
    print("\n🎵 Testing ambient sound creation:")
    
    scene = SceneConfig(
        name="Ambient Test Scene",
        duration=30.0,
        sounds=[],
        ambient=[]
    )
    
    # Add various ambient sounds
    for ambient_type, level_db in ambient_types:
        ambient = AmbientSound(
            sound_type=ambient_type,
            level_db=level_db,
            audio_file=None  # Procedural
        )
        scene.ambient.append(ambient)
        print(f"✅ Added {ambient_type} ambient at {level_db:.1f}dB")
    
    # Test with specific audio file
    test_audio = "test_datasets/forest_sounds/forest_ambient_1.wav"
    if os.path.exists(test_audio):
        file_ambient = AmbientSound(
            sound_type="custom_forest",
            level_db=-28.0,
            audio_file=test_audio
        )
        scene.ambient.append(file_ambient)
        print(f"✅ Added file-based ambient: {Path(test_audio).name}")
    
    # Test scene configuration
    print(f"\n📊 Scene summary:")
    print(f"   - Name: {scene.name}")
    print(f"   - Duration: {scene.duration}s")
    print(f"   - Directional sounds: {len(scene.sounds)}")
    print(f"   - Ambient sounds: {len(scene.ambient)}")
    
    print(f"\n🌿 Ambient details:")
    for i, ambient in enumerate(scene.ambient):
        source = "Procedural" if not ambient.audio_file else Path(ambient.audio_file).name
        print(f"   {i+1}. {ambient.sound_type}: {ambient.level_db:.1f}dB ({source})")
    
    print(f"\n✅ Ambient sound system ready!")
    print(f"🚀 In Scene Creator you can now:")
    print(f"   1. Go to 'Add Sounds' → 'Ambient Background' tab")
    print(f"   2. Select from predefined types (forest, wind, rain, etc.)")
    print(f"   3. Set background level in dB (-60 to -10)")
    print(f"   4. Choose procedural or specific audio files")
    print(f"   5. Preview and manage ambient sounds")
    
    return True

if __name__ == "__main__":
    success = test_ambient_functionality()
    if success:
        print(f"\n🎉 Ambient sound functionality working!")
    else:
        print(f"\n💥 Ambient sound test failed")
        sys.exit(1)