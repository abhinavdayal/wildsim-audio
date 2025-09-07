"""
Enhanced Wildlife Acoustic Scene Simulator
Fixes critical audio processing issues for realistic mixing
"""

import numpy as np
import pyroomacoustics as pra
import librosa
import soundfile as sf
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import random
from tqdm import tqdm

from simple_config import SceneConfig, SoundPlacement, AmbientSound
from dataset_manager import DatasetManager


class EnhancedSimulator:
    """Enhanced acoustic scene simulator with proper audio mixing"""
    
    def __init__(self, dataset_manager: DatasetManager):
        self.dataset_manager = dataset_manager
        
        # ReSpeaker USB 4 Mic Array geometry (in meters)
        self.mic_positions = np.array([
            [-0.032, 0.000, 0.000],  # Mic 1: Left
            [0.000, -0.032, 0.000],  # Mic 2: Back
            [0.032, 0.000, 0.000],   # Mic 3: Right
            [0.000, 0.032, 0.000]    # Mic 4: Front
        ]).T  # Shape: (3, 4)
    
    def load_audio_file_realistic(self, filepath: str, target_sr: int = 16000, 
                                max_duration: float = None, preserve_dynamics: bool = True) -> Tuple[np.ndarray, Dict]:
        """Load audio file with realistic dynamic range preservation"""
        try:
            audio, sr = librosa.load(filepath, sr=target_sr, mono=True)
            
            # Trim to max duration if specified
            if max_duration and len(audio) > max_duration * target_sr:
                audio = audio[:int(max_duration * target_sr)]
            
            # Calculate original dynamics info
            original_rms = np.sqrt(np.mean(audio**2)) if len(audio) > 0 else 0
            original_max = np.max(np.abs(audio)) if len(audio) > 0 else 0
            original_dynamic_range = original_max / (original_rms + 1e-10)
            
            if preserve_dynamics:
                # Gentle normalization that preserves relative loudness
                # Only normalize if signal is very quiet or very loud
                if original_max > 0:
                    if original_max < 0.01:  # Very quiet signal
                        # Boost quiet signals moderately
                        audio = audio * (0.1 / original_max)
                        print(f"  Boosted quiet signal: {original_max:.4f} → 0.1")
                    elif original_max > 0.95:  # Very loud signal  
                        # Reduce loud signals to prevent clipping
                        audio = audio * (0.8 / original_max)
                        print(f"  Reduced loud signal: {original_max:.4f} → 0.8")
                    # else: preserve original levels (most common case)
                        
            else:
                # Legacy normalization (destroys dynamics)
                if original_max > 0:
                    audio = audio / original_max
                    
            # Return audio with metadata about original dynamics
            metadata = {
                'original_rms': original_rms,
                'original_max': original_max,
                'dynamic_range': original_dynamic_range,
                'final_rms': np.sqrt(np.mean(audio**2)),
                'final_max': np.max(np.abs(audio))
            }
            
            return audio, metadata
            
        except Exception as e:
            print(f"Warning: Could not load {filepath}: {e}")
            duration = max_duration or 1.0
            silence = np.zeros(int(duration * target_sr))
            metadata = {'original_rms': 0, 'original_max': 0, 'dynamic_range': 0, 'final_rms': 0, 'final_max': 0}
            return silence, metadata
    
    def add_realistic_ambient(self, mic_signals: np.ndarray, ambient: AmbientSound, 
                            config: SceneConfig) -> np.ndarray:
        """Add ambient sounds with realistic microphone variation"""
        
        total_samples = mic_signals.shape[1]
        
        if ambient.audio_file and Path(ambient.audio_file).exists():
            try:
                # Load ambient audio
                ambient_audio, _ = self.load_audio_file_realistic(
                    ambient.audio_file, config.sample_rate, config.duration
                )
                
                # Convert dB to linear scale
                level_linear = 10**(ambient.level_db / 20)
                ambient_audio = ambient_audio * level_linear
                
                # Ensure correct length
                if len(ambient_audio) > total_samples:
                    ambient_audio = ambient_audio[:total_samples]
                elif len(ambient_audio) < total_samples:
                    repeats = (total_samples // len(ambient_audio)) + 1
                    ambient_audio = np.tile(ambient_audio, repeats)[:total_samples]
                
                # Add realistic variation between microphones
                for mic_idx in range(4):
                    # Create slight variations for each mic (simulating spatial diversity)
                    variation_factor = 1.0 + np.random.normal(0, 0.02)  # ±2% variation
                    phase_shift = np.random.randint(0, 16)  # Small time shifts
                    
                    # Apply variation
                    mic_ambient = ambient_audio * variation_factor
                    
                    # Apply small phase shift
                    if phase_shift > 0 and len(mic_ambient) > phase_shift:
                        mic_ambient = np.concatenate([
                            mic_ambient[phase_shift:], 
                            np.zeros(phase_shift)
                        ])
                    
                    mic_signals[mic_idx] += mic_ambient
                
                print(f"✓ Added realistic ambient: {Path(ambient.audio_file).name} at {ambient.level_db:.1f}dB with mic variations")
                
            except Exception as e:
                print(f"⚠️ Could not load ambient audio {ambient.audio_file}: {e}")
                # Fallback to procedural noise with variation
                self.add_procedural_ambient(mic_signals, ambient.level_db, ambient.sound_type)
        else:
            # Procedural ambient
            self.add_procedural_ambient(mic_signals, ambient.level_db, ambient.sound_type)
            
        return mic_signals
    
    def add_procedural_ambient(self, mic_signals: np.ndarray, level_db: float, sound_type: str):
        """Add procedural ambient noise with realistic microphone variation"""
        
        total_samples = mic_signals.shape[1]
        noise_level_linear = 10**(level_db / 20) * 0.2
        
        # Create correlated noise (simulating distant ambient environment)
        base_noise = np.random.normal(0, noise_level_linear, total_samples)
        
        for mic_idx in range(4):
            # Each mic gets base noise plus independent component
            independent_noise = np.random.normal(0, noise_level_linear * 0.3, total_samples)
            mic_noise = base_noise * 0.7 + independent_noise * 0.3  # Mix correlated and independent
            mic_signals[mic_idx] += mic_noise
        
        print(f"✓ Added procedural ambient: {sound_type} at {level_db:.1f}dB with mic correlation")
    
    def simulate_scene_enhanced(self, config: SceneConfig) -> Tuple[np.ndarray, Dict]:
        """
        Enhanced scene simulation with proper audio mixing
        """
        
        print(f"🎵 Enhanced simulation: {config.name}")
        print(f"Duration: {config.duration}s, Sounds: {len(config.sounds)}")
        
        # Improved room setup
        if config.sounds:
            max_distance = max([np.sqrt(sum(s.position[i]**2 for i in range(3))) for s in config.sounds])
        else:
            max_distance = 100
            
        # More realistic room sizing
        room_margin = max(50, max_distance * 2)  # Minimum 50m margin
        room_size = [room_margin * 2, room_margin * 2, 30]  # More realistic height
        
        # Forest acoustic properties (more realistic absorption)
        forest_absorption = 0.7  # Reduced from 0.99 for more realistic forest acoustics
        forest_material = pra.Material(forest_absorption)
        room = pra.ShoeBox(room_size, fs=config.sample_rate, materials=forest_material)
        
        # Center microphone array
        mic_positions_centered = self.mic_positions + np.array([
            [room_size[0]/2], [room_size[1]/2], [2]
        ])
        room.add_microphone_array(mic_positions_centered)
        
        print(f"✓ Enhanced forest environment:")
        print(f"  - Room: {room_size[0]:.0f}x{room_size[1]:.0f}x{room_size[2]:.0f}m")
        print(f"  - Absorption: {forest_absorption:.1f} (realistic forest)")
        print(f"  - Max source distance: {max_distance:.1f}m")
        
        # Initialize output
        total_samples = int(config.duration * config.sample_rate)
        mic_signals = np.zeros((4, total_samples))
        
        # Metadata
        metadata = {
            'config': config.to_dict(),
            'mic_positions': self.mic_positions.tolist(),
            'sample_rate': config.sample_rate,
            'duration': config.duration,
            'sounds': [],
            'audio_processing': {
                'dynamics_preserved': True,
                'realistic_ambient': True,
                'forest_absorption': forest_absorption
            }
        }
        
        # Process sounds with enhanced dynamics
        print("Processing sounds with preserved dynamics...")
        for i, sound in enumerate(tqdm(config.sounds, desc="Processing sounds")):
            
            # Select audio file
            audio_file = sound.audio_file
            if not audio_file or not Path(audio_file).exists():
                print(f"⚠️ Skipping sound {i}: file not found")
                continue
            
            # Load with dynamics preservation
            audio, audio_metadata = self.load_audio_file_realistic(
                audio_file, config.sample_rate, config.duration, preserve_dynamics=True
            )
            
            if len(audio) == 0:
                continue
                
            # Apply user volume scaling (now meaningful since dynamics are preserved)
            audio = audio * sound.volume
            
            # Add to pyroomacoustics
            sound_position_centered = [
                sound.position[0] + room_size[0]/2,
                sound.position[1] + room_size[1]/2, 
                sound.position[2]
            ]
            
            room.add_source(sound_position_centered, signal=audio, delay=sound.start_time)
            
            # Enhanced metadata
            sound_metadata = {
                'index': i,
                'audio_file': audio_file,
                'position': sound.position,
                'distance': sound.distance,
                'azimuth': sound.azimuth,
                'elevation': sound.elevation,
                'start_time': sound.start_time,
                'volume': sound.volume,
                'sound_type': sound.sound_type,
                'duration': len(audio) / config.sample_rate,
                'audio_dynamics': audio_metadata
            }
            metadata['sounds'].append(sound_metadata)
        
        # Run acoustic simulation
        print("Running enhanced pyroomacoustics simulation...")
        room.simulate()
        
        # Get microphone signals
        if len(room.mic_array.signals) > 0:
            mic_signals = np.array(room.mic_array.signals)
            
            # Ensure correct shape and duration
            if mic_signals.shape[0] != 4:
                print(f"⚠️ Expected 4 microphones, got {mic_signals.shape[0]}")
            
            if mic_signals.shape[1] > total_samples:
                mic_signals = mic_signals[:, :total_samples]
            elif mic_signals.shape[1] < total_samples:
                padding = total_samples - mic_signals.shape[1]
                mic_signals = np.pad(mic_signals, ((0,0), (0,padding)), 'constant')
        else:
            print("⚠️ No signals generated from sources")
            
        # Add enhanced ambient sounds
        print("Adding enhanced ambient sounds...")
        for ambient in config.ambient:
            mic_signals = self.add_realistic_ambient(mic_signals, ambient, config)
        
        # Intelligent normalization
        max_val = np.max(np.abs(mic_signals))
        if max_val > 0.98:  # Only normalize if very close to clipping
            mic_signals = mic_signals * (0.95 / max_val)
            print(f"✓ Clipping prevention: {max_val:.3f} → 0.95")
        elif max_val < 0.001:
            print(f"⚠️ Very quiet output: max={max_val:.6f} - check source levels")
        else:
            print(f"✓ Good signal levels: max={max_val:.3f} (no normalization needed)")
        
        # Final quality metrics
        rms_levels = [np.sqrt(np.mean(mic_signals[i]**2)) for i in range(4)]
        print(f"✓ Final RMS levels: {[f'{rms:.4f}' for rms in rms_levels]}")
        print(f"✓ Dynamic range preserved: {max_val / (np.mean(rms_levels) + 1e-10):.1f}")
        
        metadata['final_levels'] = {
            'max_amplitude': float(max_val),
            'rms_levels': [float(rms) for rms in rms_levels],
            'dynamic_range': float(max_val / (np.mean(rms_levels) + 1e-10))
        }
        
        print(f"✅ Enhanced simulation complete: {mic_signals.shape}")
        
        return mic_signals, metadata
        
    def save_results_enhanced(self, mic_signals: np.ndarray, metadata: Dict, output_dir: str):
        """Save enhanced simulation results with quality report"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        scene_name = metadata['config']['name'].replace(' ', '_').lower()
        
        # Save audio files
        for i in range(4):
            mic_file = output_path / f"{scene_name}_mic_{i+1}_enhanced.wav"
            sf.write(mic_file, mic_signals[i], metadata['sample_rate'])
        
        # Save enhanced metadata (convert numpy types for JSON compatibility)
        import json
        
        def convert_numpy_types(obj):
            """Convert numpy types to Python types for JSON serialization"""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
            
        # Convert metadata recursively
        def make_json_serializable(data):
            if isinstance(data, dict):
                return {key: make_json_serializable(value) for key, value in data.items()}
            elif isinstance(data, list):
                return [make_json_serializable(item) for item in data]
            else:
                return convert_numpy_types(data)
        
        json_metadata = make_json_serializable(metadata)
        
        metadata_file = output_path / f"{scene_name}_enhanced_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(json_metadata, f, indent=2)
        
        # Create quality report
        quality_report = output_path / f"{scene_name}_quality_report.txt"
        with open(quality_report, 'w') as f:
            f.write("Enhanced Audio Simulation Quality Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Scene: {metadata['config']['name']}\n")
            f.write(f"Duration: {metadata['duration']}s\n")
            f.write(f"Sample Rate: {metadata['sample_rate']}Hz\n")
            f.write(f"Sources: {len(metadata['sounds'])}\n\n")
            
            f.write("Audio Processing:\n")
            f.write(f"- Dynamics Preserved: {metadata['audio_processing']['dynamics_preserved']}\n")
            f.write(f"- Realistic Ambient: {metadata['audio_processing']['realistic_ambient']}\n")
            f.write(f"- Forest Absorption: {metadata['audio_processing']['forest_absorption']}\n\n")
            
            f.write("Final Audio Quality:\n")
            levels = metadata['final_levels']
            f.write(f"- Max Amplitude: {levels['max_amplitude']:.4f}\n")
            f.write(f"- RMS Levels: {[f'{rms:.4f}' for rms in levels['rms_levels']]}\n")
            f.write(f"- Dynamic Range: {levels['dynamic_range']:.1f}\n\n")
            
            f.write("Individual Sound Dynamics:\n")
            for sound in metadata['sounds']:
                if 'audio_dynamics' in sound:
                    dyn = sound['audio_dynamics']
                    f.write(f"- {Path(sound['audio_file']).name}:\n")
                    f.write(f"  Original RMS: {dyn['original_rms']:.4f}, Max: {dyn['original_max']:.4f}\n")
                    f.write(f"  Dynamic Range: {dyn['dynamic_range']:.1f}\n")
        
        print(f"✅ Enhanced results saved to {output_path}")
        print(f"   - Audio files: {scene_name}_mic_X_enhanced.wav")
        print(f"   - Metadata: {scene_name}_enhanced_metadata.json") 
        print(f"   - Quality report: {scene_name}_quality_report.txt")
        
        return output_path


# Comparison function to test old vs new
def compare_simulators(config: SceneConfig, dataset_manager: DatasetManager, output_dir: str = "comparison_output"):
    """Compare original vs enhanced simulator output"""
    
    from simple_simulator import SimpleSimulator
    
    print("🔄 Running comparison: Original vs Enhanced Simulator")
    print("=" * 60)
    
    # Original simulator
    print("\n1️⃣ Running Original Simulator...")
    original_sim = SimpleSimulator(dataset_manager)
    original_signals, original_meta = original_sim.simulate_scene(config)
    
    # Enhanced simulator  
    print("\n2️⃣ Running Enhanced Simulator...")
    enhanced_sim = EnhancedSimulator(dataset_manager)
    enhanced_signals, enhanced_meta = enhanced_sim.simulate_scene_enhanced(config)
    
    # Save both results
    original_sim.save_results(original_signals, original_meta, f"{output_dir}/original")
    enhanced_sim.save_results_enhanced(enhanced_signals, enhanced_meta, f"{output_dir}/enhanced")
    
    # Quality comparison
    print("\n📊 Quality Comparison:")
    print("-" * 30)
    
    orig_max = np.max(np.abs(original_signals))
    enh_max = np.max(np.abs(enhanced_signals))
    
    orig_rms = np.mean([np.sqrt(np.mean(original_signals[i]**2)) for i in range(4)])
    enh_rms = np.mean([np.sqrt(np.mean(enhanced_signals[i]**2)) for i in range(4)])
    
    print(f"Max Amplitude - Original: {orig_max:.4f}, Enhanced: {enh_max:.4f}")
    print(f"Average RMS - Original: {orig_rms:.4f}, Enhanced: {enh_rms:.4f}")
    print(f"Dynamic Range - Original: {orig_max/(orig_rms+1e-10):.1f}, Enhanced: {enh_max/(enh_rms+1e-10):.1f}")
    
    print(f"\n✅ Comparison complete! Check {output_dir}/ for results")
    
    return {
        'original': {'signals': original_signals, 'metadata': original_meta},
        'enhanced': {'signals': enhanced_signals, 'metadata': enhanced_meta}
    }