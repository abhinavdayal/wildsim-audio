"""
Demo script for testing ZODAS pipeline with simulated data
"""

import numpy as np
import soundfile as sf
from pathlib import Path
import tempfile
from zodas import ZODASPipeline


def create_test_data():
    """Create test 4-channel audio data for ZODAS testing"""
    
    # Parameters
    duration = 5.0  # seconds
    sample_rate = 16000
    n_samples = int(duration * sample_rate)
    
    # Create a chirp signal
    t = np.linspace(0, duration, n_samples)
    signal = np.sin(2 * np.pi * 220 * t) * np.exp(-0.5 * t)  # Decaying chirp
    
    # Simulate 4-microphone array with TDOA
    # Source at 45 degrees azimuth, 0 elevation
    source_azimuth = 45.0  # degrees
    source_elevation = 0.0  # degrees
    
    # Microphone positions (circular array, 6.4cm diameter)
    mic_positions = np.array([
        [0.032, 0.0, 0.0],      # Mic 1: Right
        [0.0, 0.032, 0.0],      # Mic 2: Back  
        [-0.032, 0.0, 0.0],     # Mic 3: Left
        [0.0, -0.032, 0.0]      # Mic 4: Front
    ])
    
    # Source direction
    az_rad = np.radians(source_azimuth)
    el_rad = np.radians(source_elevation)
    source_direction = np.array([
        np.cos(el_rad) * np.cos(az_rad),
        np.cos(el_rad) * np.sin(az_rad),
        np.sin(el_rad)
    ])
    
    # Speed of sound
    c = 343.0  # m/s
    
    # Create microphone signals with delays
    mic_signals = []
    for i, mic_pos in enumerate(mic_positions):
        # Calculate TDOA relative to first microphone
        if i == 0:
            delay_samples = 0
        else:
            # TDOA calculation
            tdoa = np.dot(mic_pos - mic_positions[0], source_direction) / c
            delay_samples = int(tdoa * sample_rate)
        
        # Apply delay to signal
        if delay_samples > 0:
            delayed_signal = np.concatenate([np.zeros(delay_samples), signal[:-delay_samples]])
        elif delay_samples < 0:
            delayed_signal = np.concatenate([signal[-delay_samples:], np.zeros(-delay_samples)])
        else:
            delayed_signal = signal.copy()
        
        # Add some noise
        noise = np.random.normal(0, 0.01, n_samples)
        mic_signals.append(delayed_signal + noise)
    
    return mic_signals, sample_rate, source_azimuth, source_elevation


def test_zodas_pipeline():
    """Test ZODAS pipeline with synthetic data"""
    
    print("🎯 Testing ZODAS Pipeline")
    print("=" * 40)
    
    # Generate test data
    print("📊 Generating test data...")
    mic_signals, sample_rate, true_azimuth, true_elevation = create_test_data()
    
    # Save to temporary files
    temp_files = []
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        
        for i, signal in enumerate(mic_signals):
            temp_file = temp_dir / f"mic_{i+1}.wav"
            sf.write(temp_file, signal, sample_rate)
            temp_files.append(str(temp_file))
        
        # Ground truth
        ground_truth = [{
            'position': [10.0, 10.0, 2.0],  # 10m away at 2m height
            'azimuth': true_azimuth,
            'elevation': true_elevation
        }]
        
        print(f"🎯 True source direction: Az={true_azimuth:.1f}°, El={true_elevation:.1f}°")
        
        # Run ZODAS pipeline
        print("🔄 Running ZODAS pipeline...")
        zodas = ZODASPipeline(
            sample_rate=sample_rate,
            frame_size=512,
            overlap=0.5
        )
        
        results = zodas.process_audio_files(temp_files, ground_truth)
        
        # Display results
        print("\n📈 Results:")
        print(f"Duration: {results['audio_duration']:.2f}s")
        print(f"Frames processed: {results['n_frames_processed']}")
        
        # Average DOA estimates
        frame_results = results['frame_results']
        if frame_results:
            avg_azimuth = np.mean([r['doa_azimuth'] for r in frame_results])
            avg_elevation = np.mean([r['doa_elevation'] for r in frame_results])
            
            print(f"\n🎯 Average DOA Estimates:")
            print(f"Azimuth: {avg_azimuth:.1f}° (error: {abs(avg_azimuth - true_azimuth):.1f}°)")
            print(f"Elevation: {avg_elevation:.1f}° (error: {abs(avg_elevation - true_elevation):.1f}°)")
        
        # Validation metrics
        if 'validation_metrics' in results:
            vm = results['validation_metrics']
            print(f"\n📊 Validation Metrics:")
            print(f"Mean azimuth error: {vm['errors']['mean_azimuth_error']:.2f}°")
            print(f"Mean elevation error: {vm['errors']['mean_elevation_error']:.2f}°")
            print(f"Azimuth accuracy (≤10°): {vm['accuracy_metrics']['azimuth_accuracy_10deg']:.1f}%")
            print(f"Elevation accuracy (≤10°): {vm['accuracy_metrics']['elevation_accuracy_10deg']:.1f}%")
        
        print("\n✅ ZODAS test completed!")
        
        return results


if __name__ == "__main__":
    results = test_zodas_pipeline()
