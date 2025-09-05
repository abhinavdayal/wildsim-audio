#!/usr/bin/env python3
"""
Create a tiny local audio dataset for testing when disk space is limited
"""

import sys
from pathlib import Path
import numpy as np
import soundfile as sf

sys.path.append(str(Path(__file__).parent))

from dataset_manager import DatasetManager
from models import DatasetConfig

def create_tiny_dataset():
    """Create a tiny local audio dataset for testing"""
    
    print("🎵 CREATING TINY TEST DATASET")
    print("=" * 50)
    
    # Create tiny audio samples
    test_dir = Path("./audio_cache/tiny_test")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate some simple audio files
    sample_rate = 22050  # Lower sample rate to save space
    
    # Create 5 tiny audio files
    for i in range(5):
        # Generate 2 seconds of different tones
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Different frequencies for each file
        freq = 440 + i * 110  # A4, B4, C#5, D#5, F5
        audio = 0.3 * np.sin(2 * np.pi * freq * t)
        
        # Add some variation
        if i % 2 == 0:
            # Add harmonics
            audio += 0.1 * np.sin(2 * np.pi * freq * 2 * t)
        else:
            # Add some noise
            audio += 0.05 * np.random.normal(0, 1, len(audio))
        
        # Save as WAV file
        filename = test_dir / f"test_audio_{i+1}_{int(freq)}hz.wav"
        sf.write(filename, audio, sample_rate)
        
        print(f"✓ Created {filename.name} ({duration}s, {freq}Hz)")
    
    print(f"\n📁 Created {len(list(test_dir.glob('*.wav')))} test audio files")
    
    # Calculate total size
    total_size = sum(f.stat().st_size for f in test_dir.glob('*.wav'))
    print(f"💾 Total size: {total_size/1024:.1f} KB")
    
    return test_dir

def test_tiny_dataset():
    """Test with the tiny local dataset"""
    
    # Create the tiny dataset
    dataset_path = create_tiny_dataset()
    
    print(f"\n🔍 TESTING DATASET MANAGER WITH TINY DATASET")
    print("=" * 50)
    
    # Create dataset config for local dataset
    config = DatasetConfig(
        dataset_name="tiny_test",
        source_type="local",
        path_or_id=str(dataset_path),
        audio_extensions=[".wav"],
        cache_dir=str(dataset_path.parent)
    )
    
    # Initialize dataset manager
    manager = DatasetManager()
    
    # Clear any existing catalog for this dataset
    if config.dataset_name in manager.catalog.catalog["datasets"]:
        print("Removing existing catalog entry...")
        del manager.catalog.catalog["datasets"][config.dataset_name]
        
        files_to_remove = []
        for file_hash, file_info in manager.catalog.catalog["files"].items():
            if file_info.get("dataset") == config.dataset_name:
                files_to_remove.append(file_hash)
        
        for file_hash in files_to_remove:
            del manager.catalog.catalog["files"][file_hash]
    
    # Setup dataset (should catalog the files)
    success = manager.setup_dataset(config)
    
    if success:
        print("✅ Setup completed successfully!")
        
        # Get summary
        summary = manager.get_dataset_summary()
        
        print(f"\n📊 DATASET SUMMARY:")
        print(f"   Total datasets: {summary['total_datasets']}")
        print(f"   Total files: {summary['total_files']}")
        
        if summary['source_type_counts']:
            print(f"\n🏷️  Source types found:")
            for source_type, count in summary['source_type_counts'].items():
                print(f"     {source_type}: {count} files")
        
        if summary['format_counts']:
            print(f"\n🎵 File formats:")
            for format_ext, count in summary['format_counts'].items():
                print(f"     {format_ext}: {count} files")
        
        # Test finding specific audio files
        print(f"\n🔎 TESTING AUDIO FILE SEARCH:")
        
        matches = manager.catalog.find_audio_files(min_duration=1.0, max_duration=3.0)
        print(f"   Audio files (1-3 seconds): {len(matches)}")
        
        if matches:
            print(f"   Found files:")
            for match in matches:
                duration = match.get('duration', 0)
                source_type = match.get('source_type', 'unknown')
                file_path = Path(match.get('absolute_path', ''))
                print(f"     🎵 {file_path.name} ({duration:.1f}s, {source_type})")
    
    else:
        print("❌ Setup failed!")

if __name__ == "__main__":
    try:
        test_tiny_dataset()
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Install with: pip install soundfile")
