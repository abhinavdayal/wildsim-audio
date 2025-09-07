"""
Simplified Wildlife Acoustic Scene Simulator
Uses real audio data and pyroomacoustics for realistic propagation
"""

import numpy as np
import pyroomacoustics as pra
import librosa
import soundfile as sf
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import random
from tqdm import tqdm

from simple_config import SceneConfig, SoundPlacement, AmbientSound, SceneBuilder
from dataset_manager import DatasetManager

class SimpleSimulator:
    """Simplified acoustic scene simulator"""
    
    def __init__(self, dataset_manager: DatasetManager):
        self.dataset_manager = dataset_manager
        self.scene_builder = SceneBuilder(dataset_manager)
        
        # ReSpeaker USB 4 Mic Array geometry (in meters)
        self.mic_positions = np.array([
            [-0.032, 0.000, 0.000],  # Mic 1: Left
            [0.000, -0.032, 0.000],  # Mic 2: Back
            [0.032, 0.000, 0.000],   # Mic 3: Right
            [0.000, 0.032, 0.000]    # Mic 4: Front
        ]).T  # Shape: (3, 4)
    
    def load_audio_file(self, filepath: str, target_sr: int = 16000, max_duration: float = None) -> np.ndarray:
        """Load and preprocess audio file"""
        try:
            audio, sr = librosa.load(filepath, sr=target_sr, mono=True)
            
            # Trim to max duration if specified
            if max_duration and len(audio) > max_duration * target_sr:
                audio = audio[:int(max_duration * target_sr)]
            
            # Normalize
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio))
            
            return audio
            
        except Exception as e:
            print(f"Warning: Could not load {filepath}: {e}")
            # Return silence if audio fails to load
            duration = max_duration or 1.0
            return np.zeros(int(duration * target_sr))
    
    def select_audio_for_type(self, sound_type: str, dataset: str = None) -> str:
        """Select a random audio file matching the sound type"""
        matches = self.scene_builder.find_audio_by_type(sound_type, dataset)
        
        if matches:
            return random.choice(matches)
        else:
            print(f"Warning: No audio found for type '{sound_type}' in dataset '{dataset}'")
            # Try without dataset restriction
            matches = self.scene_builder.find_audio_by_type(sound_type)
            if matches:
                return random.choice(matches)
            else:
                print(f"Warning: No audio found for type '{sound_type}' at all")
                return None
    
    def select_random_from_dataset(self, dataset: str) -> str:
        """Select a random audio file from a specific dataset"""
        matches = []
        for file_hash, file_info in self.dataset_manager.catalog.catalog["files"].items():
            if file_info.get("dataset") == dataset:
                matches.append(file_info["absolute_path"])
        
        if matches:
            return random.choice(matches)
        else:
            print(f"Warning: No audio files found in dataset '{dataset}'")
            return None
    
    def simulate_scene(self, config: SceneConfig) -> Tuple[np.ndarray, Dict]:
        """
        Simulate acoustic scene and return microphone array signals
        
        Returns:
            mic_signals: (4, samples) array of microphone signals
            metadata: Dictionary with ground truth information
        """
        
        print(f"Simulating scene: {config.name}")
        print(f"Duration: {config.duration}s, Sounds: {len(config.sounds)}")
        
        # Create anechoic (wall-less) forest environment
        # For outdoor forest environments, we want no reflections from walls
        # Make room very large with maximum absorption to simulate open air
        max_distance = max([np.sqrt(sum(s.position[i]**2 for i in range(3))) for s in config.sounds], default=100)
        room_size = [max_distance * 5, max_distance * 5, 300]  # Very large for minimal boundary effects
        
        # Use maximum absorption (0.99) to simulate anechoic forest environment
        # This minimizes wall reflections to simulate open-air propagation
        forest_material = pra.Material(0.99)  # Nearly perfect absorption = no reflections
        room = pra.ShoeBox(room_size, fs=config.sample_rate, materials=forest_material)
        
        # Move microphone array to center of room  
        # Position at forest floor level (2m above ground) for realistic forest recording
        mic_positions_centered = self.mic_positions + np.array([[room_size[0]/2], [room_size[1]/2], [2]])  # 2m above forest floor
        
        # Add microphone array - positioned for forest environment
        room.add_microphone_array(mic_positions_centered)
        
        # Forest acoustic properties
        print(f"✓ Forest environment configured:")
        print(f"  - Anechoic simulation (no wall reflections)")
        print(f"  - Room size: {room_size[0]:.0f}x{room_size[1]:.0f}x{room_size[2]:.0f}m")
        print(f"  - Microphones at forest level: 2m above ground")
        print(f"  - Material absorption: 99% (open air)")
        
        # Set forest-specific atmospheric conditions
        # Forest environments typically have:
        # - Higher humidity (affects sound absorption)
        # - Variable temperature (affects sound speed)  
        # - Natural barriers (trees, vegetation) - simulated via high absorption
        # These are implicitly modeled through the high absorption coefficient
        
        # Prepare output
        total_samples = int(config.duration * config.sample_rate)
        mic_signals = np.zeros((4, total_samples))
        
        # Metadata for ground truth
        metadata = {
            'config': config.to_dict(),
            'mic_positions': self.mic_positions.tolist(),
            'sample_rate': config.sample_rate,
            'duration': config.duration,
            'sounds': []
        }
        
        # Process each sound
        for i, sound in enumerate(tqdm(config.sounds, desc="Processing sounds")):
            
            # Select audio file if not specified or if random selection requested
            audio_file = sound.audio_file
            
            if audio_file and audio_file.startswith("random:"):
                # Handle random selection from specific dataset
                dataset_name = audio_file.split(":", 1)[1]
                audio_file = self.select_random_from_dataset(dataset_name)
                if not audio_file:
                    continue
            elif not audio_file or not Path(audio_file).exists():
                # Try to find audio by sound type (fallback)
                audio_file = self.select_audio_for_type(sound.sound_type)
                if not audio_file:
                    continue
            
            # Load audio
            audio = self.load_audio_file(audio_file, config.sample_rate, config.duration)
            if len(audio) == 0:
                continue
            
            # Apply volume scaling
            audio = audio * sound.volume
            
            # Adjust sound position to room coordinates (center at room center)
            sound_position_centered = [
                sound.position[0] + room_size[0]/2,
                sound.position[1] + room_size[1]/2, 
                sound.position[2]
            ]
            
            # Add sound source to room
            room.add_source(sound_position_centered, signal=audio, delay=sound.start_time)
            
            # Store metadata
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
                'duration': len(audio) / config.sample_rate
            }
            metadata['sounds'].append(sound_metadata)
        
        # Run simulation
        print("Running pyroomacoustics simulation...")
        room.simulate()
        
        # Get microphone signals
        if len(room.mic_array.signals) > 0:
            mic_signals = np.array(room.mic_array.signals)
            
            # Ensure correct shape (4, samples)
            if mic_signals.shape[0] != 4:
                print(f"Warning: Expected 4 microphones, got {mic_signals.shape[0]}")
            
            # Trim to exact duration
            if mic_signals.shape[1] > total_samples:
                mic_signals = mic_signals[:, :total_samples]
            elif mic_signals.shape[1] < total_samples:
                # Pad with zeros if too short
                padding = total_samples - mic_signals.shape[1]
                mic_signals = np.pad(mic_signals, ((0,0), (0,padding)), 'constant')
        else:
            print("Warning: No signals generated")
        
        # Add ambient sounds (using actual audio files, not white noise)
        for ambient in config.ambient:
            if ambient.audio_file and Path(ambient.audio_file).exists():
                try:
                    # Load ambient audio file
                    ambient_audio = self.load_audio_file(ambient.audio_file, config.sample_rate, config.duration)
                    
                    # Convert dB level to linear scale
                    level_linear = 10**(ambient.level_db / 20)
                    ambient_audio = ambient_audio * level_linear
                    
                    # Ensure ambient audio is the right length
                    if len(ambient_audio) > total_samples:
                        ambient_audio = ambient_audio[:total_samples]
                    elif len(ambient_audio) < total_samples:
                        # Loop the ambient audio to fill the scene duration
                        repeats = (total_samples // len(ambient_audio)) + 1
                        ambient_audio = np.tile(ambient_audio, repeats)[:total_samples]
                    
                    # Add ambient audio to all microphones (non-directional)
                    for mic_idx in range(4):
                        mic_signals[mic_idx] += ambient_audio
                        
                    print(f"✓ Added ambient: {Path(ambient.audio_file).name} at {ambient.level_db:.1f}dB")
                        
                except Exception as e:
                    print(f"⚠️  Could not load ambient audio {ambient.audio_file}: {e}")
                    # Fallback to low-level noise only if audio fails
                    noise_level_linear = 10**(ambient.level_db / 20) * 0.1  # Much quieter fallback
                    noise = np.random.normal(0, noise_level_linear, (4, total_samples))
                    mic_signals += noise
            else:
                # Procedural ambient (very low level background)
                noise_level_linear = 10**(ambient.level_db / 20) * 0.2  # Quiet procedural background
                noise = np.random.normal(0, noise_level_linear, (4, total_samples))
                mic_signals += noise
                print(f"✓ Added procedural ambient: {ambient.sound_type} at {ambient.level_db:.1f}dB")
        
        # Gentle normalization to prevent clipping while preserving volume differences
        max_val = np.max(np.abs(mic_signals))
        if max_val > 0.95:
            # Only normalize if very close to clipping, preserving relative volumes
            mic_signals = mic_signals * (0.95 / max_val)
            print(f"✓ Gentle normalization (clipping prevention): max level {max_val:.3f} → 0.95")
        elif max_val < 0.001:
            # Handle very quiet signals
            print(f"⚠️ Very quiet signal: max level {max_val:.6f} - check volume settings")
        else:
            print(f"✓ No normalization needed: max level {max_val:.3f} (preserving volume differences)")
        
        # Calculate final SNR for verification
        signal_power = np.mean(np.abs(mic_signals))
        print(f"✓ Final signal level: {signal_power:.4f} (linear), {20*np.log10(signal_power):.1f}dB")
        
        print(f"✓ Simulation complete. Output shape: {mic_signals.shape}")
        
        return mic_signals, metadata
    
    def save_results(self, mic_signals: np.ndarray, metadata: Dict, output_dir: str):
        """Save simulation results"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        scene_name = metadata['config']['name'].replace(' ', '_').lower()
        
        # Save audio files
        for i in range(4):
            mic_file = output_path / f"{scene_name}_mic_{i+1}.wav"
            sf.write(mic_file, mic_signals[i], metadata['sample_rate'])
        
        # Save metadata
        import json
        metadata_file = output_path / f"{scene_name}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Results saved to {output_path}")
        return output_path

def load_config_from_yaml(yaml_file: str, dataset_manager: DatasetManager) -> SceneConfig:
    """Load scene configuration from YAML file"""
    import yaml
    
    with open(yaml_file, 'r') as f:
        data = yaml.safe_load(f)
    
    scene_builder = SceneBuilder(dataset_manager)
    
    # Create sound placements
    sounds = []
    for sound_config in data.get('sounds', []):
        # Select audio file based on type and dataset
        audio_file = scene_builder.select_audio_for_type(
            sound_config['sound_type'], 
            sound_config.get('dataset')
        )
        
        if audio_file:
            sound = scene_builder.place_sound_at_angle(
                audio_file=audio_file,
                distance=sound_config['distance'],
                azimuth_deg=sound_config['azimuth'],
                elevation_deg=sound_config.get('elevation', 0.0),
                start_time=sound_config['start_time'],
                volume=sound_config.get('volume', 1.0),
                sound_type=sound_config['sound_type']
            )
            sounds.append(sound)
    
    # Create ambient conditions
    ambient = [
        AmbientSound(
            sound_type=a['sound_type'],
            level_db=a['level_db'],
            audio_file=a.get('audio_file')
        )
        for a in data.get('ambient', [])
    ]
    
    return SceneConfig(
        name=data['name'],
        duration=data['duration'],
        sounds=sounds,
        ambient=ambient,
        sample_rate=data.get('sample_rate', 16000),
        description=data.get('description', '')
    )

# Example usage
if __name__ == "__main__":
    # Initialize dataset manager
    dm = DatasetManager()
    
    # Load a config
    config = load_config_from_yaml("configs/simple_elephant.yaml", dm)
    
    # Create simulator
    sim = SimpleSimulator(dm)
    
    # Run simulation
    mic_signals, metadata = sim.simulate_scene(config)
    
    # Save results
    sim.save_results(mic_signals, metadata, "outputs/simple_simulation")