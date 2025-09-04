"""
Wildlife Acoustic Scene Simulation Library using Pyroomacoustics
Generates realistic synthetic data for microphone array algorithm validation
"""

import numpy as np
import pyroomacoustics as pra
import librosa
import soundfile as sf
from typing import List, Dict, Tuple, Optional, Union
import json
import os
from pathlib import Path
import scipy.signal as signal
from tqdm import tqdm

# Import all Pydantic models from models.py
from models import (
    MicrophonePosition, SoundSource, EnvironmentConfig, 
    AmbientCondition, SimulationConfig, ProcessingConfig,
    SourceType, EnvironmentType
)

class SoundLibrary:
    """Enhanced sound library with better audio handling"""
    
    def __init__(self):
        self.loaded_sounds = {}
        self.sound_info = {}  # Store metadata about sounds
    
    def load_audio(self, file_path: str, target_sr: int = 16000) -> Tuple[np.ndarray, Dict]:
        """Load audio file with metadata"""
        cache_key = f"{file_path}_{target_sr}"
        
        if cache_key not in self.loaded_sounds:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Audio file not found: {file_path}")
            
            # Load with librosa for better audio processing
            audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
            
            # Store audio and metadata
            self.loaded_sounds[cache_key] = audio
            self.sound_info[cache_key] = {
                'duration': len(audio) / target_sr,
                'sample_rate': target_sr,
                'original_file': file_path,
                'peak_amplitude': np.max(np.abs(audio)),
                'rms_level': np.sqrt(np.mean(audio**2))
            }
        
        return self.loaded_sounds[cache_key].copy(), self.sound_info[cache_key]
    
    def get_sound_info(self, file_path: str, target_sr: int = 16000) -> Dict:
        """Get metadata about a sound file"""
        cache_key = f"{file_path}_{target_sr}"
        if cache_key in self.sound_info:
            return self.sound_info[cache_key]
        
        # Load just to get info
        self.load_audio(file_path, target_sr)
        return self.sound_info[cache_key]

class AmbientGenerator:
    """Enhanced ambient sound generation"""
    
    def __init__(self, sample_rate: int = 16000):
        self.sr = sample_rate
    
    def generate_rain(self, duration: float, intensity: str = 'moderate') -> np.ndarray:
        """Generate realistic rain sound using advanced filtering"""
        num_samples = int(duration * self.sr)
        
        # Generate multiple layers of rain
        intensity_params = {
            'light': {'density': 0.3, 'droplet_rate': 50, 'amplitude': 0.2},
            'moderate': {'density': 0.5, 'droplet_rate': 100, 'amplitude': 0.4}, 
            'heavy': {'density': 0.8, 'droplet_rate': 200, 'amplitude': 0.7}
        }
        
        params = intensity_params.get(intensity, intensity_params['moderate'])
        
        # Create base noise
        base_noise = np.random.normal(0, 0.1, num_samples)
        
        # Add droplet impacts (impulsive components)
        droplet_times = np.random.poisson(params['droplet_rate'], int(duration))
        droplets = np.zeros(num_samples)
        
        for i, count in enumerate(droplet_times):
            if i * self.sr < num_samples:
                for _ in range(count):
                    pos = int(i * self.sr + np.random.uniform(0, self.sr))
                    if pos < num_samples:
                        # Create droplet impact
                        impact_dur = int(0.01 * self.sr)  # 10ms impact
                        impact = np.exp(-np.linspace(0, 5, impact_dur))
                        end_pos = min(pos + impact_dur, num_samples)
                        droplets[pos:end_pos] += impact[:end_pos-pos] * np.random.uniform(0.5, 1.5)
        
        # Combine and filter
        rain = base_noise + droplets * params['density']
        
        # Apply spectral shaping for realistic rain sound
        sos_high = signal.butter(3, 500, btype='high', fs=self.sr, output='sos')
        sos_low = signal.butter(2, 8000, btype='low', fs=self.sr, output='sos')
        
        rain = signal.sosfilt(sos_high, rain)
        rain = signal.sosfilt(sos_low, rain)
        
        return rain * params['amplitude']
    
    def generate_wind(self, duration: float, speed_kmh: float = 15) -> np.ndarray:
        """Generate realistic wind sound with gusts"""
        num_samples = int(duration * self.sr)
        
        # Generate colored noise (brown/red noise for low frequencies)
        white = np.random.normal(0, 1, num_samples)
        
        # Create wind spectrum (emphasis on low frequencies)
        freqs = np.fft.rfftfreq(num_samples, 1/self.sr)
        freqs[0] = 1  # Avoid division by zero
        
        # Wind has 1/f^2 characteristics (brown noise) at low frequencies
        wind_filter = 1 / (freqs**1.5 + 10)  # Modified to avoid excessive low-freq
        
        spectrum = np.fft.rfft(white)
        wind_spectrum = spectrum * wind_filter
        wind_base = np.fft.irfft(wind_spectrum, n=num_samples)
        
        # Add gusts (amplitude modulation)
        gust_freq = 0.1  # Hz (10-second gusts)
        gust_envelope = 1 + 0.3 * np.sin(2 * np.pi * gust_freq * np.linspace(0, duration, num_samples))
        
        # Add turbulence (higher frequency modulation)
        turbulence = 1 + 0.1 * np.sin(2 * np.pi * 2 * np.linspace(0, duration, num_samples))
        
        # Scale with wind speed
        wind_scale = np.clip(speed_kmh / 30.0, 0.1, 2.0)
        
        # Apply filtering for natural wind sound
        sos = signal.butter(4, [20, 2000], btype='band', fs=self.sr, output='sos')
        wind = signal.sosfilt(sos, wind_base * gust_envelope * turbulence)
        
        return wind * wind_scale * 0.3
    
    def generate_forest_base(self, duration: float) -> np.ndarray:
        """Generate subtle forest ambient with distant sounds"""
        num_samples = int(duration * self.sr)
        
        # Very low-level ambient
        ambient = np.random.normal(0, 0.05, num_samples)
        
        # Add occasional distant sounds (very subtle)
        for _ in range(int(duration * 0.5)):  # Random events
            pos = np.random.randint(0, num_samples - int(2 * self.sr))
            
            # Create distant sound (could be very faint animal calls)
            distant_dur = np.random.uniform(0.5, 2.0)
            distant_samples = int(distant_dur * self.sr)
            
            freq = np.random.uniform(200, 800)  # Distant low-freq sounds
            distant_sound = np.sin(2 * np.pi * freq * np.linspace(0, distant_dur, distant_samples))
            distant_sound *= np.exp(-np.linspace(0, 3, distant_samples))  # Decay
            distant_sound *= np.random.uniform(0.01, 0.03)  # Very quiet
            
            end_pos = min(pos + distant_samples, num_samples)
            ambient[pos:end_pos] += distant_sound[:end_pos-pos]
        
        # Apply gentle filtering
        sos = signal.butter(2, 1000, btype='low', fs=self.sr, output='sos')
        ambient = signal.sosfilt(sos, ambient)
        
        return ambient * 0.15

class ForestAcousticEnvironment:
    """Enhanced acoustic environment modeling using pyroomacoustics"""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.room = None
        self.sound_library = SoundLibrary()
        self.ambient_gen = AmbientGenerator(config.sample_rate)
        
        self._create_environment()
    
    def _create_environment(self):
        """Create the acoustic environment (forest clearing)"""
        env = self.config.environment
        
        # Create outdoor "room" with forest-like acoustic properties
        # Use very low absorption to simulate outdoor propagation
        materials = pra.Material(env.absorption_coefficient)
        
        self.room = pra.ShoeBox(
            env.room_size,
            fs=self.config.sample_rate,
            materials=materials,
            max_order=env.max_order,  # Limited reflections for outdoor
            air_absorption=env.air_absorption,
            humidity=env.humidity,
            temperature=env.temperature
        )
        
        # Add microphone array with exact ReSpeaker geometry
        self._add_microphone_array()
    
    def _add_microphone_array(self):
        """Add the microphone array to the room"""
        # Convert microphone positions to numpy array
        mic_positions = np.array([
            [mic.x, mic.y, mic.z] for mic in self.config.mic_positions
        ]).T  # Shape: (3, n_mics)
        
        # Translate to center of room for better simulation
        room_center = np.array(self.config.environment.room_size) / 2
        mic_positions += room_center.reshape(-1, 1)
        
        # Create microphone array
        self.room.add_microphone_array(
            pra.MicrophoneArray(mic_positions, fs=self.config.sample_rate)
        )
    
    def add_sound_sources(self) -> List[Dict]:
        """Add all sound sources to the environment"""
        source_metadata = []
        
        for source in self.config.sources:
            try:
                # Load audio
                audio_data, audio_info = self.sound_library.load_audio(
                    source.audio_file, self.config.sample_rate
                )
                
                # Apply volume scaling
                audio_data = audio_data * source.volume_scale
                
                # Handle duration
                if source.duration_override:
                    target_samples = int(source.duration_override * self.config.sample_rate)
                    if len(audio_data) > target_samples:
                        audio_data = audio_data[:target_samples]
                    else:
                        # Repeat if too short
                        repeats = int(np.ceil(target_samples / len(audio_data)))
                        audio_data = np.tile(audio_data, repeats)[:target_samples]
                
                # Calculate position relative to room center
                room_center = np.array(self.config.environment.room_size) / 2
                source_pos = np.array(source.position) + room_center
                
                # Ensure source is within room bounds
                source_pos = np.clip(source_pos, [1, 1, 1], 
                                   np.array(self.config.environment.room_size) - [1, 1, 1])
                
                # Add source to room
                self.room.add_source(source_pos, signal=audio_data, delay=source.start_time)
                
                # Store metadata
                metadata = {
                    'original_position': source.position,
                    'room_position': source_pos.tolist(),
                    'audio_info': audio_info,
                    'source_config': source.__dict__
                }
                source_metadata.append(metadata)
                
            except Exception as e:
                print(f"Warning: Could not add source {source.audio_file}: {e}")
                source_metadata.append({'error': str(e)})
        
        return source_metadata
    
    def simulate(self) -> Tuple[np.ndarray, Dict]:
        """Run the acoustic simulation"""
        # Add sources
        source_metadata = self.add_sound_sources()
        
        # Run pyroomacoustics simulation
        self.room.simulate(snr=None)  # No SNR limitation
        
        # Get microphone signals
        mic_signals = self.room.mic_array.signals
        
        # Ensure we have the right duration
        target_samples = int(self.config.scene_duration * self.config.sample_rate)
        if mic_signals.shape[1] < target_samples:
            # Pad with zeros if too short
            padding = target_samples - mic_signals.shape[1]
            mic_signals = np.pad(mic_signals, ((0, 0), (0, padding)), 'constant')
        elif mic_signals.shape[1] > target_samples:
            # Truncate if too long
            mic_signals = mic_signals[:, :target_samples]
        
        # Add ambient sounds
        mic_signals = self._add_ambient_sounds(mic_signals)
        
        # Add background noise
        mic_signals = self._add_background_noise(mic_signals)
        
        # Create comprehensive metadata
        metadata = self._create_metadata(source_metadata, mic_signals)
        
        return mic_signals, metadata
    
    def _add_ambient_sounds(self, mic_signals: np.ndarray) -> np.ndarray:
        """Add ambient environmental sounds"""
        for condition_type, condition in self.config.ambient.items():
            if condition_type == 'rain':
                rain = self.ambient_gen.generate_rain(
                    self.config.scene_duration, condition.intensity
                )
                # Rain affects all mics similarly with slight variations
                rain_level = 10**(condition.level_db / 20) * 0.01
                for i in range(mic_signals.shape[0]):
                    variation = np.random.normal(1.0, 0.05)
                    mic_signals[i] += rain * rain_level * variation
            
            elif condition_type == 'wind':
                wind = self.ambient_gen.generate_wind(
                    self.config.scene_duration, condition.intensity
                )
                wind_level = 10**(condition.level_db / 20) * 0.01
                for i in range(mic_signals.shape[0]):
                    variation = np.random.normal(1.0, 0.1)  # More variation for wind
                    mic_signals[i] += wind * wind_level * variation
            
            elif condition_type == 'forest_base':
                forest = self.ambient_gen.generate_forest_base(self.config.scene_duration)
                forest_level = 10**(condition.level_db / 20) * 0.01
                for i in range(mic_signals.shape[0]):
                    variation = np.random.normal(1.0, 0.03)
                    mic_signals[i] += forest * forest_level * variation
        
        return mic_signals
    
    def _add_background_noise(self, mic_signals: np.ndarray) -> np.ndarray:
        """Add uncorrelated background noise"""
        noise_amplitude = 10**(self.config.background_noise_level / 20) * 0.005
        
        for i in range(mic_signals.shape[0]):
            noise = np.random.normal(0, noise_amplitude, mic_signals.shape[1])
            mic_signals[i] += noise
        
        return mic_signals
    
    def _create_metadata(self, source_metadata: List[Dict], 
                        mic_signals: np.ndarray) -> Dict:
        """Create comprehensive simulation metadata"""
        # Calculate ground truth using pyroomacoustics tools
        ground_truth = self._compute_ground_truth()
        
        metadata = {
            'config': {
                'scene_duration': self.config.scene_duration,
                'sample_rate': self.config.sample_rate,
                'frame_size': self.config.frame_size,
                'hop_length': self.config.hop_length,
                'environment': self.config.environment.__dict__,
                'background_noise_level': self.config.background_noise_level
            },
            'sources': source_metadata,
            'mic_positions': [
                {'x': mic.x, 'y': mic.y, 'z': mic.z} 
                for mic in self.config.mic_positions
            ],
            'mic_signals_stats': {
                'shape': mic_signals.shape,
                'peak_amplitude': float(np.max(np.abs(mic_signals))),
                'rms_level': float(np.sqrt(np.mean(mic_signals**2))),
                'dynamic_range': float(20 * np.log10(np.max(np.abs(mic_signals)) / 
                                                   (np.sqrt(np.mean(mic_signals**2)) + 1e-10)))
            },
            'ground_truth': ground_truth,
            'pyroomacoustics_info': {
                'room_size': self.room.shoebox_dim,
                'absorption_coefficient': self.config.environment.absorption_coefficient,
                'max_order': self.config.environment.max_order,
                'n_sources': len(self.room.sources),
                'n_mics': self.room.mic_array.n_mics
            }
        }
        
        return metadata
    
    def _compute_ground_truth(self) -> Dict:
        """Compute ground truth DOA and propagation information"""
        ground_truth = {
            'source_directions': [],
            'propagation_delays': [],
            'distances': [],
            'source_powers': []
        }
        
        # Get microphone array center for reference
        mic_center = np.mean(self.room.mic_array.R, axis=1)
        
        for i, source in enumerate(self.room.sources):
            source_pos = source.position
            
            # Calculate direction from mic array center
            direction_vector = source_pos - mic_center
            distance = np.linalg.norm(direction_vector)
            
            # Calculate azimuth and elevation
            azimuth = np.arctan2(direction_vector[1], direction_vector[0]) * 180 / np.pi
            if azimuth < 0:
                azimuth += 360
            
            elevation = np.arcsin(direction_vector[2] / distance) * 180 / np.pi
            
            # Calculate propagation delays to each microphone
            mic_delays = []
            mic_distances = []
            
            for mic_idx in range(self.room.mic_array.n_mics):
                mic_pos = self.room.mic_array.R[:, mic_idx]
                mic_distance = np.linalg.norm(source_pos - mic_pos)
                delay = mic_distance / pra.constants.get('c')  # Speed of sound
                
                mic_delays.append(delay)
                mic_distances.append(mic_distance)
            
            # Calculate TDOAs between mic pairs
            tdoas = []
            for j in range(self.room.mic_array.n_mics):
                for k in range(j+1, self.room.mic_array.n_mics):
                    tdoa = mic_delays[k] - mic_delays[j]
                    tdoas.append({
                        'mic_pair': (j, k),
                        'tdoa_seconds': tdoa,
                        'tdoa_samples': tdoa * self.config.sample_rate
                    })
            
            ground_truth['source_directions'].append({
                'azimuth_deg': azimuth,
                'elevation_deg': elevation,
                'position': source_pos.tolist(),
                'distance_from_center': distance
            })
            
            ground_truth['propagation_delays'].append({
                'mic_delays_seconds': mic_delays,
                'mic_distances': mic_distances,
                'tdoas': tdoas
            })
            
            ground_truth['distances'].append(mic_distances)
            ground_truth['source_powers'].append(np.mean(source.signal**2))
        
        return ground_truth

class WildlifeAcousticSimulator:
    """Main simulator class using pyroomacoustics"""
    
    def __init__(self):
        self.environments = {}  # Cache for reused environments
    
    def simulate_scene(self, config: SimulationConfig) -> Tuple[np.ndarray, Dict]:
        """
        Simulate a complete acoustic scene
        
        Returns:
            mic_signals: (n_mics, n_samples) array of microphone signals
            metadata: Comprehensive simulation metadata including ground truth
        """
        # Create environment
        environment = ForestAcousticEnvironment(config)
        
        # Run simulation
        mic_signals, metadata = environment.simulate()
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(mic_signals))
        if max_val > 0.95:
            mic_signals = mic_signals * (0.95 / max_val)
            metadata['normalization_applied'] = True
            metadata['normalization_factor'] = 0.95 / max_val
        
        return mic_signals, metadata
    
    def validate_with_doa_algorithms(self, mic_signals: np.ndarray, 
                                   config: SimulationConfig) -> Dict:
        """
        Validate simulation using built-in pyroomacoustics DOA algorithms
        """
        # Create microphone array for DOA estimation
        mic_positions = np.array([
            [mic.x, mic.y, mic.z] for mic in config.mic_positions
        ]).T
        
        mic_array = pra.MicrophoneArray(mic_positions, fs=config.sample_rate)
        
        validation_results = {}
        
        try:
            # MUSIC algorithm
            music = pra.doa.MUSIC(mic_array, fs=config.sample_rate, nfft=config.frame_size)
            music.locate_sources(mic_signals, num_src=len(config.sources))
            
            validation_results['MUSIC'] = {
                'azimuth_estimates': music.azimuth_recon.tolist() if hasattr(music, 'azimuth_recon') else [],
                'spatial_spectrum': music.grid.values.tolist() if hasattr(music, 'grid') else []
            }
        except Exception as e:
            validation_results['MUSIC'] = {'error': str(e)}
        
        try:
            # SRP-PHAT algorithm (matches your approach)
            srp = pra.doa.SRP(mic_array, fs=config.sample_rate, nfft=config.frame_size)
            srp.locate_sources(mic_signals, num_src=len(config.sources))
            
            validation_results['SRP-PHAT'] = {
                'azimuth_estimates': srp.azimuth_recon.tolist() if hasattr(srp, 'azimuth_recon') else [],
                'spatial_spectrum': srp.grid.values.tolist() if hasattr(srp, 'grid') else []
            }
        except Exception as e:
            validation_results['SRP-PHAT'] = {'error': str(e)}
        
        try:
            # FRIDA (Frequency Invariant Beamforming)
            frida = pra.doa.FRIDA(mic_array, fs=config.sample_rate, nfft=config.frame_size)
            frida.locate_sources(mic_signals, num_src=len(config.sources))
            
            validation_results['FRIDA'] = {
                'azimuth_estimates': frida.azimuth_recon.tolist() if hasattr(frida, 'azimuth_recon') else [],
                'spatial_spectrum': frida.grid.values.tolist() if hasattr(frida, 'grid') else []
            }
        except Exception as e:
            validation_results['FRIDA'] = {'error': str(e)}
        
        return validation_results

# Utility functions
def save_simulation_results(mic_signals: np.ndarray, metadata: Dict, 
                          output_dir: str, scene_id: str = "scene_001"):
    """Enhanced save function with more formats"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save as WAV files for audio compatibility
    for i, channel in enumerate(mic_signals):
        wav_path = output_path / f"{scene_id}_mic_{i+1}.wav"
        sf.write(wav_path, channel, metadata['config']['sample_rate'])
    
    # Save as numpy array for processing
    np.save(output_path / f"{scene_id}_all_channels.npy", mic_signals)
    
    # Save metadata as JSON
    with open(output_path / f"{scene_id}_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    print(f"Simulation results saved to {output_path}")
    print(f"- Audio files: {scene_id}_mic_[1-4].wav")  
    print(f"- Array data: {scene_id}_all_channels.npy")
    print(f"- Metadata: {scene_id}_metadata.json")

def create_example_config() -> SimulationConfig:
    """Create example configuration for testing"""
    
    sources = [
        SoundSource(
            audio_file="sounds/elephant_call.wav",
            position=(100.0, 0.0, 0.0),  # 100m North
            start_time=2.0,
            volume_scale=1.0,
            source_type="elephant"
        ),
        SoundSource(
            audio_file="sounds/bird_call.wav", 
            position=(50.0, 50.0, 8.0),  # 50m NE, elevated
            start_time=0.5,
            volume_scale=0.7,
            source_type="bird"
        )
    ]
    
    ambient = {
        'wind': AmbientCondition('wind', 12.0, 28.0),
        'forest_base': AmbientCondition('forest_base', 'medium', 25.0)
    }
    
    return SimulationConfig(
        scene_duration=10.0,
        sample_rate=16000,
        sources=sources,
        ambient=ambient,
        background_noise_level=45.0,
        environment=EnvironmentConfig(
            room_size=(1000.0, 1000.0, 50.0),  # 1km forest clearing
            absorption_coefficient=0.02,
            max_order=2,
            temperature=25.0,
            humidity=60.0
        )
    )

if __name__ == "__main__":
    # Example usage
    simulator = WildlifeAcousticSimulator()
    
    try:
        # Create example configuration
        config = create_example_config()
        
        print("Wildlife Acoustic Simulation with Pyroomacoustics")
        print("=" * 50)
        print(f"Scene duration: {config.scene_duration}s")
        print(f"Sample rate: {config.sample_rate} Hz")
        print(f"Sources: {len(config.sources)}")
        print(f"Environment: {config.environment.room_size}")
        
        # Run simulation
        mic_signals, metadata = simulator.simulate_scene(config)
        
        print(f"\nSimulation completed!")
        print(f"Output shape: {mic_signals.shape}")
        print(f"Peak amplitude: {metadata['mic_signals_stats']['peak_amplitude']:.3f}")
        print(f"Dynamic range: {metadata['mic_signals_stats']['dynamic_range']:.1f} dB")
        
        # Validate with built-in DOA algorithms
        validation = simulator.validate_with_doa_algorithms(mic_signals, config)
        print(f"\nDOA Validation completed:")
        for method, results in validation.items():
            if 'error' not in results:
                print(f"- {method}: Found {len(results.get('azimuth_estimates', []))} sources")
            else:
                print(f"- {method}: Error - {results['error']}")
        
        # Save results
        save_simulation_results(mic_signals, metadata, "output", "example_scene")
        
        print(f"\nGround Truth Information:")
        for i, direction in enumerate(metadata['ground_truth']['source_directions']):
            print(f"Source {i+1}: {direction['azimuth_deg']:.1f}° at {direction['distance_from_center']:.1f}m")
        
    except FileNotFoundError as e:
        print(f"Audio files not found: {e}")
        print("\nTo run with real audio files:")
        print("1. Create 'sounds/' directory")
        print("2. Add elephant_call.wav, bird_call.wav, etc.")
        print("3. Run the simulation")
        
        print("\nAlternatively, run the demo script with synthetic audio!")
        
    except Exception as e:
        print(f"Simulation error: {e}")
        print("Check your pyroomacoustics installation and configuration")