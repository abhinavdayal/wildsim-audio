#!/usr/bin/env python3
"""
Example usage of the scene generator for elephant detection training data
"""

import subprocess
import sys
from pathlib import Path

def generate_training_dataset():
    """Generate a training dataset for elephant detection"""
    
    print("🐘 Generating Elephant Detection Training Dataset")
    print("=" * 50)
    
    # Configuration
    num_samples = 100
    positive_ratio = 0.4  # 40% positive samples (contains elephants)
    
    print(f"📊 Dataset Configuration:")
    print(f"   - Total samples: {num_samples}")
    print(f"   - Positive samples (with elephants): {int(num_samples * positive_ratio)}")
    print(f"   - Negative samples (no elephants): {int(num_samples * (1 - positive_ratio))}")
    print(f"   - Directional sounds per scene: 10-50")
    print(f"   - Ambient sounds per scene: 3-5")
    print(f"   - Distance range: 10-500 meters")
    print(f"   - Scene duration: 30 seconds")
    
    # Generate scenes
    cmd = [
        "uv", "run", "python", "scene_generator.py",
        "--num-samples", str(num_samples),
        "--positive-ratio", str(positive_ratio),
        "--min-directional", "10",
        "--max-directional", "50", 
        "--min-ambient", "3",
        "--max-ambient", "5",
        "--duration", "30.0",
        "--output-dir", "training_data"
    ]
    
    print(f"\n🚀 Running scene generation...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        
        # Extract JSONL path from output
        jsonl_path = None
        for line in result.stdout.split('\n'):
            if 'JSONL file:' in line:
                jsonl_path = line.split('JSONL file: ')[1].strip()
                break
        
        if jsonl_path:
            print(f"\n✅ Scene generation completed!")
            print(f"📁 JSONL configuration file: {jsonl_path}")
            
            # Ask if user wants to render audio
            response = input(f"\n🎵 Would you like to render audio for all {num_samples} scenes? (y/N): ").strip().lower()
            
            if response in ['y', 'yes']:
                render_audio(jsonl_path)
            else:
                print(f"\n💡 To render audio later, run:")
                print(f"   uv run python scene_generator.py --render --jsonl-path {jsonl_path}")
                
        else:
            print("❌ Could not find JSONL path in output")
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during generation: {e}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)

def render_audio(jsonl_path):
    """Render 4-mic audio files from JSONL configuration"""
    
    output_dir = Path(jsonl_path).parent / "rendered_audio"
    
    print(f"\n🎵 Rendering 4-microphone audio files...")
    print(f"📁 Output directory: {output_dir}")
    
    cmd = [
        "uv", "run", "python", "scene_generator.py",
        "--render",
        "--jsonl-path", jsonl_path,
        "--render-output", str(output_dir)
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        print(f"\n✅ Audio rendering completed!")
        print(f"🎧 4-channel audio files saved to: {output_dir}")
        print(f"📊 Each scene contains 4 WAV files (mic_1.wav, mic_2.wav, mic_3.wav, mic_4.wav)")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during rendering: {e}")
        if e.stderr:
            print("STDERR:", e.stderr)

def generate_small_test_dataset():
    """Generate a small test dataset for quick validation"""
    
    print("🧪 Generating Small Test Dataset")
    print("=" * 35)
    
    # Small test configuration
    num_samples = 10
    positive_ratio = 0.5
    
    cmd = [
        "uv", "run", "python", "scene_generator.py",
        "--num-samples", str(num_samples),
        "--positive-ratio", str(positive_ratio),
        "--min-directional", "5",
        "--max-directional", "15",
        "--min-ambient", "2", 
        "--max-ambient", "4",
        "--output-dir", "test_data"
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        print(f"\n✅ Test dataset generated!")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        generate_small_test_dataset()
    else:
        generate_training_dataset()