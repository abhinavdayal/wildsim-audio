#!/usr/bin/env python3
"""
Test Enhanced Audio Quality
Compare original vs enhanced simulator to verify fixes
"""

from enhanced_simulator import EnhancedSimulator, compare_simulators
from simple_config import SceneConfig, SoundPlacement, AmbientSound
from dataset_manager import DatasetManager
import numpy as np
from pathlib import Path

def create_test_scene(dataset_manager: DatasetManager) -> SceneConfig:
    """Create a test scene for quality comparison"""
    
    # Find some test audio files
    elephant_files = []
    bird_files = []
    
    # Search for audio files
    for file_hash, file_info in dataset_manager.catalog.catalog.get("files", {}).items():
        abs_path = file_info["absolute_path"]
        if "elephant" in abs_path.lower():
            elephant_files.append(abs_path)
        elif "bird" in abs_path.lower():
            bird_files.append(abs_path)
    
    # Create test sounds with different expected volume levels
    sounds = []
    
    if elephant_files:
        # Loud elephant call (should be naturally loud)
        sounds.append(SoundPlacement(
            audio_file=elephant_files[0],
            position=(80.0, 45.0, 1.5),  # 80m away, NE direction
            start_time=1.0,
            volume=1.0,  # Natural volume
            sound_type="elephant"
        ))
    
    if bird_files:
        # Quiet bird chirp (should be naturally quieter)
        sounds.append(SoundPlacement(
            audio_file=bird_files[0],
            position=(25.0, -30.0, 3.0),  # 25m away, bird in tree
            start_time=3.0,
            volume=1.0,  # Natural volume - should be quieter than elephant
            sound_type="bird"
        ))
        
        # Distant bird (even quieter)
        if len(bird_files) > 1:
            sounds.append(SoundPlacement(
                audio_file=bird_files[1] if len(bird_files) > 1 else bird_files[0],
                position=(150.0, 90.0, 5.0),  # Far away bird
                start_time=5.0,
                volume=0.7,  # Reduced volume + distance = very quiet
                sound_type="bird"
            ))
    
    # Ambient forest sounds
    ambient = [
        AmbientSound(
            sound_type="forest_background",
            level_db=-40.0  # Quiet background
        )
    ]
    
    return SceneConfig(
        name="Audio Quality Test Scene",
        duration=10.0,
        sounds=sounds,
        ambient=ambient,
        sample_rate=16000,
        description="Test scene for comparing audio quality between simulators"
    )

def analyze_audio_quality(signals: np.ndarray, name: str):
    """Analyze audio quality metrics"""
    
    print(f"\n📊 Audio Quality Analysis: {name}")
    print("-" * 40)
    
    # Basic stats
    max_val = np.max(np.abs(signals))
    rms_levels = [np.sqrt(np.mean(signals[i]**2)) for i in range(4)]
    avg_rms = np.mean(rms_levels)
    
    print(f"Max amplitude: {max_val:.4f}")
    print(f"Average RMS: {avg_rms:.4f}")
    print(f"Dynamic range: {max_val/(avg_rms+1e-10):.1f}")
    
    # Per-microphone analysis
    print(f"RMS per mic: {[f'{rms:.4f}' for rms in rms_levels]}")
    
    # Quality indicators
    if max_val > 0.98:
        print("⚠️  Near clipping - may have distortion")
    elif max_val < 0.001:
        print("⚠️  Very quiet - may lack presence")
    else:
        print("✅ Good amplitude levels")
        
    if max(rms_levels) / (min(rms_levels) + 1e-10) > 2.0:
        print("⚠️  Large RMS variation between mics")
    else:
        print("✅ Consistent RMS between microphones")
        
    return {
        'max_amplitude': max_val,
        'avg_rms': avg_rms,
        'rms_levels': rms_levels,
        'dynamic_range': max_val/(avg_rms+1e-10)
    }

if __name__ == "__main__":
    print("🧪 Audio Quality Test: Original vs Enhanced Simulator")
    print("=" * 60)
    
    # Initialize
    dm = DatasetManager()
    
    # Create test scene
    print("🏗️  Creating test scene...")
    test_scene = create_test_scene(dm)
    
    print(f"Test scene created:")
    print(f"- Sounds: {len(test_scene.sounds)}")
    print(f"- Duration: {test_scene.duration}s")
    print(f"- Ambient: {len(test_scene.ambient)} conditions")
    
    # Run comparison
    results = compare_simulators(test_scene, dm, "audio_quality_test")
    
    # Detailed analysis
    original_quality = analyze_audio_quality(results['original']['signals'], "Original Simulator")
    enhanced_quality = analyze_audio_quality(results['enhanced']['signals'], "Enhanced Simulator")
    
    # Comparison summary
    print("\n🔍 Quality Comparison Summary:")
    print("-" * 40)
    
    print(f"Dynamic Range:")
    print(f"  Original: {original_quality['dynamic_range']:.1f}")
    print(f"  Enhanced: {enhanced_quality['dynamic_range']:.1f}")
    
    print(f"Average RMS:")
    print(f"  Original: {original_quality['avg_rms']:.4f}")
    print(f"  Enhanced: {enhanced_quality['avg_rms']:.4f}")
    
    # Recommendations
    print("\n💡 Recommendations:")
    if enhanced_quality['dynamic_range'] > original_quality['dynamic_range']:
        print("✅ Enhanced simulator preserves better dynamic range")
    else:
        print("⚠️  Original simulator may have better dynamic range")
        
    if abs(enhanced_quality['avg_rms'] - 0.1) < abs(original_quality['avg_rms'] - 0.1):
        print("✅ Enhanced simulator has better target RMS level")
    else:
        print("⚠️  Original simulator may have better RMS level")
    
    print(f"\n📁 Test results saved to: audio_quality_test/")
    print("   Listen to both versions and compare quality!")
    
    print("\n🎧 Manual listening test:")
    print("afplay audio_quality_test/original/audio_quality_test_scene_mic_1.wav")  
    print("afplay audio_quality_test/enhanced/audio_quality_test_scene_mic_1_enhanced.wav")