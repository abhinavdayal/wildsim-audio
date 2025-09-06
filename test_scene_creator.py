#!/usr/bin/env python3
"""
Simple test script for Scene Creator functionality
Tests the core simulation pipeline end-to-end
"""

import os
import sys
from pathlib import Path
import numpy as np

# Add current directory to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent))

from simple_config import SceneConfig, SoundPlacement, AmbientSound
from simple_simulator import SimpleSimulator
from dataset_manager import DatasetManager

def create_test_dataset():
    """Create a simple test dataset using scipy test files"""
    print("🔧 Setting up test dataset...")
    
    # Initialize dataset manager
    dataset_manager = DatasetManager()
    
    # Find a test audio file from scipy
    test_files = [
        ".venv/lib/python3.12/site-packages/scipy/io/tests/data/test-44100Hz-2ch-32bit-float-le.wav",
        ".venv/lib/python3.12/site-packages/scipy/io/tests/data/test-44100Hz-le-1ch-4bytes.wav"
    ]
    
    # Use the first available test file
    test_file = None
    for file_path in test_files:
        if os.path.exists(file_path):
            test_file = os.path.abspath(file_path)
            print(f"✅ Found test audio file: {test_file}")
            break
    
    if not test_file:
        raise FileNotFoundError("No test audio files found")
    
    return dataset_manager, test_file

def create_simple_scene(audio_file):
    """Create a simple test scene with one sound source"""
    print("🎬 Creating test scene...")
    
    # Create a simple scene configuration
    scene = SceneConfig(
        name="Test Scene",
        description="Simple test with one sound source",
        duration=2.0,  # Short 2-second test
        sounds=[
            SoundPlacement(
                audio_file=audio_file,
                position=(100.0, 50.0, 0.0),  # 100m away, 50m to the right
                start_time=0.0,
                volume=1.0,
                sound_type="test_audio"
            )
        ],
        ambient=[
            AmbientSound("forest", -45.0)  # Quiet forest background
        ]
    )
    
    print(f"✅ Created scene: {scene.name}")
    print(f"   - Duration: {scene.duration}s")
    print(f"   - Sound sources: {len(scene.sounds)}")
    print(f"   - First source at: {scene.sounds[0].position}")
    
    return scene

def test_simulation(dataset_manager, scene):
    """Test the simulation pipeline"""
    print("🚀 Running simulation test...")
    
    try:
        # Create simulator
        simulator = SimpleSimulator(dataset_manager)
        
        # Run simulation
        print("   - Initializing simulator...")
        mic_signals, metadata = simulator.simulate_scene(scene)
        
        print("✅ Simulation completed successfully!")
        print(f"   - Output shape: {mic_signals.shape}")
        print(f"   - Sample rate: {metadata['sample_rate']} Hz")
        print(f"   - Duration: {mic_signals.shape[1] / metadata['sample_rate']:.2f}s")
        print(f"   - Max amplitude: {np.max(np.abs(mic_signals)):.3f}")
        
        # Basic sanity checks
        assert mic_signals.shape[0] == 4, f"Expected 4 mics, got {mic_signals.shape[0]}"
        assert mic_signals.shape[1] > 0, "No audio samples generated"
        assert metadata['sample_rate'] > 0, "Invalid sample rate"
        
        print("✅ All simulation checks passed!")
        return True
        
    except Exception as e:
        print(f"❌ Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the complete test"""
    print("🧪 Testing Scene Creator functionality...\n")
    
    try:
        # Step 1: Setup
        dataset_manager, test_audio = create_test_dataset()
        
        # Step 2: Create scene
        scene = create_simple_scene(test_audio)
        
        # Step 3: Test simulation
        success = test_simulation(dataset_manager, scene)
        
        if success:
            print("\n🎉 Scene Creator test PASSED!")
            print("✅ The simulation pipeline is working correctly")
        else:
            print("\n💥 Scene Creator test FAILED!")
            return 1
            
    except Exception as e:
        print(f"\n💥 Test setup failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)