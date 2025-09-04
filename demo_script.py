"""
Demo script for acoustic scene simulation using synthetic audio signals
This allows testing without requiring actual audio files
"""

import numpy as np
import matplotlib.pyplot as plt
from acoustic_scene_generator import *
from config_builder import SceneConfigBuilder
import scipy.signal as signal
from pathlib import Path
import tempfile
import os

class SyntheticAudioGenerator:
    """Generate synthetic audio signals for different source types"""
    
    def __init__(self, sample_rate: int = 16000):
        self.sr = sample_rate
    
    def generate_elephant_call(self, duration: float = 3.0) -> np.ndarray:
        """Generate synthetic elephant rumble (low frequency)"""
        t = np.linspace(0, duration, int(duration * self.sr), False)
        
        # Low frequency rumble (5-100 Hz dominant)
        fundamental = 12  # Hz
        signal = (np.sin(2 * np.pi * fundamental * t) * 0.6 +
                 np.sin(2 * np.pi * fundamental * 2 * t) * 0.3 +
                 np.sin(2 * np.pi * fundamental * 3 * t) * 0.1)
        
        # Add some formant-like structure
        envelope = np.exp(-t * 0.8)  # Decay
        signal = signal * envelope
        
        # Add frequency modulation for naturalness
        fm = 0.1 * np.sin(2 * np.pi * 2 * t)
        signal = signal * (1 + fm)
        
        # Apply low-pass filter to emphasize low frequencies
        sos = signal.butter(4, 150, btype='low', fs=self.sr, output='sos')
        signal = signal.sosfilt(sos, signal)
        
        return signal * 0.5
    
    def generate_bird_call(self, duration: float = 2.0) -> np.ndarray:
        """Generate synthetic bird call (higher frequency chirps)"""
        t = np.linspace(0, duration, int(duration * self.sr), False)
        
        # Create chirping pattern
        n_chirps = 3
        chirp_duration = duration / n_chirps
        signal = np.zeros_like(t)
        
        for i in range(n_chirps):
            start_idx = int(i * chirp_duration * self.sr)
            end_idx = int((i + 0.7) * chirp_duration * self.sr)
            
            if end_idx > len(t):
                end_idx = len(t)
            
            chirp_t = t[start_idx:end_idx] - t[start_idx]
            
            # Frequency sweep from 2kHz to 4kHz
            f0, f1 = 2000, 4000
            chirp = signal.chirp(chirp_t, f0, chirp_t[-1], f1, method='linear')
            
            # Apply envelope
            envelope = np.sin(np.pi * chirp_t / chirp_t[-1])**2
            signal[start_idx:end_idx] = chirp * envelope * 0.3
        
        return signal
    
    def generate_vehicle_sound(self, duration: float = 5.0) -> np.ndarray:
        """Generate synthetic vehicle engine sound"""
        t = np.linspace(0, duration, int(duration * self.sr), False)
        
        # Engine rumble - lower frequencies
        engine_freq = 80  # Hz
        engine = (np.sin(2 * np.pi * engine_freq * t) * 0.4 +
                 np.sin(2 * np.pi * engine_freq * 1.5 * t) * 0.3 +
                 np.sin(2 * np.pi * engine_freq * 2 * t) * 0.2)
        
        # Add road noise - broadband
        road_noise = np.random.normal(0, 0.1, len(t))
        
        # Band-pass filter for tire/road interaction
        sos = signal.butter(2, [200, 2000], btype='band', fs=self.sr, output='sos')
        road_noise = signal.sosfilt(sos, road_noise)
        
        # Combine and add slow amplitude modulation
        vehicle_sound = engine + road_noise * 0.5
        modulation = 1 + 0.1 * np.sin(2 * np.pi * 3 * t)  # 3 Hz modulation
        
        return vehicle_sound * modulation * 0.3
    
    def generate_monkey_call(self, duration: float = 2.5) -> np.ndarray:
        """Generate synthetic monkey vocalization"""
        t = np.linspace(0, duration, int(duration * self.sr), False)
        
        # Create chattering/barking pattern
        n_calls = 5
        signal = np.zeros_like(t)
        
        for i in range(n_calls):
            start_time = i * duration / n_calls + np.random.uniform(0, 0.1)
            start_idx = int(start_time * self.sr)
            call_duration = 0.3
            end_idx = min(start_idx + int(call_duration * self.sr), len(t))
            
            if start_idx >= len(t):
                break
                
            call_t = t[start_idx:end_idx] - t[start_idx]
            
            # Bark-like sound with harmonics
            fundamental = 400 + np.random.uniform(-50, 50)
            call = (np.sin(2 * np.pi * fundamental * call_t) * 0.6 +
                   np.sin(2 * np.pi * fundamental * 2 * call_t) * 0.3 +
                   np.sin(2 * np.pi * fundamental * 3 * call_t) * 0.1)
            
            # Apply envelope
            envelope = np.exp(-call_t * 8) * np.sin(np.pi * call_t / call_t[-1])**0.5
            signal[start_idx:end_idx] = call * envelope * 0.4
        
        return signal

class DemoRunner:
    """Runs demonstration scenarios with synthetic audio"""
    
    def __init__(self):
        self.audio_gen = SyntheticAudioGenerator()
        self.temp_dir = tempfile.mkdtemp()
        print(f"Using temporary directory: {self.temp_dir}")
    
    def create_synthetic_audio_files(self):
        """Create temporary synthetic audio files"""
        
        # Generate different types of sounds
        sounds = {
            'elephant_call.wav': self.audio_gen.generate_elephant_call(3.0),
            'bird_call.wav': self.audio_gen.generate_bird_call(2.0), 
            'vehicle.wav': self.audio_gen.generate_vehicle_sound(5.0),
            'monkey_call.wav': self.audio_gen.generate_monkey_call(2.5),
            'elephant_adult.wav': self.audio_gen.generate_elephant_call(4.0),
            'elephant_calf.wav': self.audio_gen.generate_elephant_call(2.0) * 0.7,
        }
        
        # Save as numpy arrays (our library loads with librosa, but for demo we'll modify)
        for filename, audio_data in sounds.items():
            filepath = os.path.join(self.temp_dir, filename)
            np.save(filepath.replace('.wav', '.npy'), audio_data)
        
        return list(sounds.keys())
    
    def modify_sound_library(self):
        """Modify the SoundLibrary class to load our synthetic audio"""
        
        def load_synthetic_audio(self, file_path: str, target_sr: int = 16000):
            """Modified load function that uses our synthetic audio"""
            cache_key = f"{file_path}_{target_sr}"
            
            if cache_key not in self.loaded_sounds:
                # Convert .wav to .npy path
                npy_path = os.path.join(self.temp_dir, 
                                       os.path.basename(file_path).replace('.wav', '.npy'))
                
                if not os.path.exists(npy_path):
                    # Create a default sound if not found
                    print(f"Creating default sound for {file_path}")
                    audio = self.audio_gen.generate_elephant_call(3.0)
                else:
                    audio = np.load(npy_path)
                
                self.loaded_sounds[cache_key] = audio
            
            return self.loaded_sounds[cache_key].copy()
        
        # Monkey patch the SoundLibrary class
        SoundLibrary.load_audio = load_synthetic_audio
        SoundLibrary.audio_gen = self.audio_gen
        SoundLibrary.temp_dir = self.temp_dir
    
    def run_simple_demo(self):
        """Run a simple demonstration with one elephant"""
        
        print("=== Simple Elephant Detection Demo ===")
        
        # Create configuration
        config = (SceneConfigBuilder()
                 .add_elephant(distance=100, azimuth_deg=45, start_time=2.0)
                 .set_forest_ambient(level_db=20.0)
                 .set_wind(speed_kmh=10, level_db=25.0)
                 .set_duration(8.0)
                 .set_noise_level(40.0)
                 .build())
        
        config.preview()
        
        # Generate scene
        generator = AcousticSceneGenerator()
        mic_signals, metadata = generator.generate_scene(config)
        
        print(f"\nGenerated scene:")
        print(f"- Shape: {mic_signals.shape}")
        print(f"- Duration: {mic_signals.shape[1] / config.sample_rate:.1f}s")
        print(f"- Peak amplitude: {np.max(np.abs(mic_signals)):.3f}")
        
        self.visualize_results(mic_signals, metadata, "Simple Demo")
        
        return mic_signals, metadata
    
    def run_complex_demo(self):
        """Run a complex demonstration with multiple sources"""
        
        print("\n=== Complex Multi-Source Demo ===")
        
        config = (SceneConfigBuilder()
                 .add_elephant(distance=150, azimuth_deg=30, start_time=3.0)
                 .add_bird(distance=40, azimuth_deg=120, elevation=8.0, start_time=1.0)
                 .add_vehicle(distance=200, azimuth_deg=270, start_time=0.5, volume=0.6)
                 .add_custom_source("monkey", distance=60, azimuth_deg=180, 
                                   elevation=12.0, start_time=5.0, 
                                   audio_file="monkey_call.wav", volume=0.8)
                 .set_rain(intensity="light", level_db=30.0)
                 .set_wind(speed_kmh=15, level_db=28.0)
                 .set_forest_ambient(level_db=25.0)
                 .set_duration(12.0)
                 .set_noise_level(45.0)
                 .build())
        
        config.preview()
        
        # Generate scene
        generator = AcousticSceneGenerator()
        mic_signals, metadata = generator.generate_scene(config)
        
        print(f"\nGenerated complex scene:")
        print(f"- Shape: {mic_signals.shape}")
        print(f"- Sources: {len(metadata['sources'])}")
        print(f"- Ground truth directions: {len(metadata['ground_truth']['source_directions'])}")
        
        self.visualize_results(mic_signals, metadata, "Complex Demo")
        
        # Show ground truth information
        self.show_ground_truth(metadata)
        
        return mic_signals, metadata
    
    def visualize_results(self, mic_signals: np.ndarray, metadata: Dict, title: str):
        """Visualize the simulation results"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=14)
        
        sample_rate = metadata['sample_rate']
        time = np.arange(mic_signals.shape[1]) / sample_rate
        
        # Plot 1: Time domain signals
        for i in range(4):
            axes[0, 0].plot(time, mic_signals[i], 
                           label=f'Mic {i+1}', alpha=0.7, linewidth=0.8)
        
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Amplitude')
        axes[0, 0].set_title('Microphone Array Signals')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Spectrogram of first microphone
        f, t_spec, Sxx = signal.spectrogram(mic_signals[0], sample_rate, 
                                           nperseg=512, noverlap=256)
        im = axes[0, 1].pcolormesh(t_spec, f, 10 * np.log10(Sxx + 1e-10), 
                                  shading='gouraud', cmap='viridis')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Frequency (Hz)')
        axes[0, 1].set_title('Spectrogram (Mic 1)')
        plt.colorbar(im, ax=axes[0, 1], label='Power (dB)')
        
        # Plot 3: Cross-correlation example (Mic 1 vs Mic 2)
        max_lag_ms = 5  # 5ms max lag
        max_lag_samples = int(max_lag_ms * sample_rate / 1000)
        
        # Compute cross-correlation for a short segment
        segment_length = min(8192, mic_signals.shape[1])
        seg1 = mic_signals[0, :segment_length]
        seg2 = mic_signals[1, :segment_length]
        
        correlation = np.correlate(seg2, seg1, mode='full')
        lags = np.arange(-len(seg1)+1, len(seg1))
        
        # Focus on reasonable lag range
        center = len(lags) // 2
        lag_range = slice(center - max_lag_samples, center + max_lag_samples + 1)
        
        axes[1, 0].plot(lags[lag_range] / sample_rate * 1000, 
                       correlation[lag_range])
        axes[1, 0].set_xlabel('Lag (ms)')
        axes[1, 0].set_ylabel('Cross-correlation')
        axes[1, 0].set_title('Cross-correlation (Mic 1 vs Mic 2)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Scene layout
        axes[1, 1].set_aspect('equal')
        
        # Plot microphones
        mic_x = [mic['x'] for mic in metadata['mic_positions']]
        mic_y = [mic['y'] for mic in metadata['mic_positions']]
        axes[1, 1].scatter(mic_x, mic_y, c='red', s=100, marker='s', 
                          label='Microphones', zorder=5)
        axes[1, 1].plot([0], [0], 'ro', markersize=8, label='Array Center')
        
        # Plot sources
        colors = ['blue', 'green', 'orange', 'purple', 'brown']
        for i, source in enumerate(metadata['sources']):
            x, y, z = source['position']
            color = colors[i % len(colors)]
            axes[1, 1].scatter(x, y, c=color, s=150, marker='o', 
                              alpha=0.8, edgecolor='black', linewidth=1,
                              label=source.get('source_type', f'Source {i+1}'))
        
        # Add range circles
        for radius in [50, 100, 200]:
            circle = plt.Circle((0, 0), radius, fill=False, 
                               linestyle='--', alpha=0.3, color='gray')
            axes[1, 1].add_patch(circle)
        
        axes[1, 1].set_xlabel('X Position (m)')
        axes[1, 1].set_ylabel('Y Position (m)')
        axes[1, 1].set_title('Scene Layout')
        axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def show_ground_truth(self, metadata: Dict):
        """Display ground truth information for validation"""
        
        print("\n=== Ground Truth Information ===")
        
        gt = metadata['ground_truth']
        
        for i, direction in enumerate(gt['source_directions']):
            source_meta = metadata['sources'][i]
            print(f"\nSource {i+1} ({source_meta.get('source_type', 'unknown')}):")
            print(f"  Position: ({source_meta['position'][0]:.0f}, "
                  f"{source_meta['position'][1]:.0f}, {source_meta['position'][2]:.0f}) m")
            print(f"  Azimuth: {direction['azimuth_deg']:.1f}°")
            print(f"  Range: {gt['source_ranges'][i]:.1f} m")
            
            print(f"  Expected TDOAs (samples):")
            for tdoa_info in gt['expected_tdoas'][i]:
                pair = tdoa_info['mic_pair']
                tdoa_samples = tdoa_info['tdoa_samples']
                print(f"    Mic {pair[0]+1}-{pair[1]+1}: {tdoa_samples:.2f} samples "
                      f"({tdoa_samples/metadata['sample_rate']*1000:.3f} ms)")
    
    def cleanup(self):
        """Clean up temporary files"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        print(f"Cleaned up temporary directory: {self.temp_dir}")

def main():
    """Run the complete demonstration"""
    
    demo = DemoRunner()
    
    try:
        # Setup synthetic audio
        print("Creating synthetic audio files...")
        audio_files = demo.create_synthetic_audio_files()
        print(f"Created {len(audio_files)} synthetic audio files")
        
        # Modify the sound library to use our synthetic audio
        demo.modify_sound_library()
        
        # Run demonstrations
        print("\n" + "="*50)
        mic_signals1, metadata1 = demo.run_simple_demo()
        
        print("\n" + "="*50)
        mic_signals2, metadata2 = demo.run_complex_demo()
        
        print("\n" + "="*50)
        print("Demo completed successfully!")
        print("\nThis demonstrates:")
        print("1. Synthetic acoustic scene generation")
        print("2. Multiple source types and positions")
        print("3. Ambient environmental conditions")
        print("4. Ground truth generation for validation")
        print("5. Visualization of results")
        
        print("\nNext steps:")
        print("1. Replace synthetic audio with real animal recordings")
        print("2. Implement your ZODAS processing pipeline")
        print("3. Compare detected DOAs with ground truth")
        print("4. Generate large training datasets")
        
    finally:
        # Clean up
        demo.cleanup()

if __name__ == "__main__":
    main()
