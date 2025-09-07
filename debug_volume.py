#!/usr/bin/env python3
"""
Debug Volume Test - Test audio volume outside Streamlit
Run this script to verify if volume multiplication works outside the web interface
"""

import numpy as np
import librosa
import soundfile as sf
import tempfile
import sys
from pathlib import Path

def test_volume_multiplication(audio_file_path: str, volumes: list = [0.1, 1.0, 3.0]):
    """Test volume multiplication with actual audio files"""
    
    if not Path(audio_file_path).exists():
        print(f"❌ Audio file not found: {audio_file_path}")
        return False
    
    print(f"🎵 Testing volume multiplication with: {Path(audio_file_path).name}")
    print(f"📊 Test volumes: {volumes}")
    print("-" * 60)
    
    try:
        # Load audio (5 seconds for testing)
        audio_data, sr = librosa.load(audio_file_path, sr=16000, duration=5.0)
        print(f"✅ Loaded audio: {len(audio_data)} samples at {sr}Hz")
        
        # Original stats
        orig_rms = np.sqrt(np.mean(audio_data**2))
        orig_max = np.max(np.abs(audio_data))
        print(f"📈 Original - RMS: {orig_rms:.6f}, Max: {orig_max:.6f}")
        print()
        
        # Test each volume
        output_files = []
        for vol in volumes:
            # Apply volume
            volume_adjusted = audio_data * vol
            
            # Calculate stats
            adj_rms = np.sqrt(np.mean(volume_adjusted**2))
            adj_max = np.max(np.abs(volume_adjusted))
            rms_ratio = adj_rms / (orig_rms + 1e-10)
            max_ratio = adj_max / (orig_max + 1e-10)
            
            print(f"🔊 Volume {vol:.1f}:")
            print(f"   - Adjusted RMS: {adj_rms:.6f} (ratio: {rms_ratio:.3f})")
            print(f"   - Adjusted Max: {adj_max:.6f} (ratio: {max_ratio:.3f})")
            
            # Verify multiplication worked
            expected_ratio = vol
            if abs(rms_ratio - expected_ratio) > 0.001:
                print(f"   ⚠️  RMS ratio mismatch! Expected: {expected_ratio:.3f}, Got: {rms_ratio:.3f}")
            else:
                print(f"   ✅ RMS ratio correct!")
            
            # Save to file for external playback
            with tempfile.NamedTemporaryFile(suffix=f'_vol_{vol:.1f}.wav', delete=False) as f:
                sf.write(f.name, volume_adjusted, sr)
                output_files.append(f.name)
                print(f"   💾 Saved to: {f.name}")
            
            print()
        
        print("🎧 MANUAL LISTENING TEST:")
        print("Play the generated files with any audio player (not Streamlit)")
        print("The volume differences should be clearly audible!")
        print()
        for i, (vol, file_path) in enumerate(zip(volumes, output_files)):
            print(f"   {i+1}. Volume {vol:.1f}: {file_path}")
        
        print()
        print("🔬 DIAGNOSIS:")
        print("- If files sound different in external player: Streamlit/browser issue")
        print("- If files sound identical in external player: Audio system issue") 
        print("- Check your system audio settings for auto-normalization/leveling")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def find_sample_audio():
    """Find a sample audio file to test with"""
    
    # Common audio locations in the project
    search_paths = [
        "audio_cache",
        "data",
        "samples",
        "test_data"
    ]
    
    audio_extensions = ['.wav', '.mp3', '.flac', '.ogg']
    
    for search_path in search_paths:
        if Path(search_path).exists():
            for ext in audio_extensions:
                audio_files = list(Path(search_path).rglob(f'*{ext}'))
                if audio_files:
                    return str(audio_files[0])
    
    return None

if __name__ == "__main__":
    print("🧪 Volume Multiplication Debug Test")
    print("=" * 60)
    
    # Get audio file from command line or find automatically
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
    else:
        print("🔍 Searching for sample audio file...")
        audio_file = find_sample_audio()
        
        if not audio_file:
            print("❌ No audio file found. Usage:")
            print("   python debug_volume.py /path/to/audio/file.wav")
            sys.exit(1)
        
        print(f"📁 Found: {audio_file}")
        print()
    
    # Run the test
    success = test_volume_multiplication(audio_file)
    
    if success:
        print("\n✅ Test completed! Check the generated files with an external audio player.")
    else:
        print("\n❌ Test failed!")