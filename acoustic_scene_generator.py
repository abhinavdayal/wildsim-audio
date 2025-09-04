"""
Acoustic Scene Simulation Library for Wildlife Monitoring
Generates synthetic data matching ReSpeaker USB 4 Mic Array configuration
"""

import numpy as np
import librosa
import scipy.signal as signal
from scipy.spatial.distance import cdist
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Union
import json
import os
from pathlib import Path

@dataclass
class MicrophonePosition:
    """Microphone position in 3D space (meters)"""
    x: float
    y: float
    z: float = 0.0

@dataclass
class SoundSource:
    """Sound source definition"""
    audio_file: str
    position: Tuple[float, float, float]  # (x, y, z) in meters
    start_time: float = 0.0
    volume_scale: float = 1.0
    source_type: str = "unknown"

@dataclass
class AmbientCondition:
    """Ambient environmental condition"""
    condition_type: str  # 'rain', 'wind', 'forest_base'
    intensity: Union[str, float]  # 'light'/'moderate'/'heavy' or numeric
    level_db: float

@dataclass
class SimulationConfig:
    """Complete simulation configuration"""
    # Scene parameters
    scene_duration: float = 10.0
    sample_rate: int = 16000
    
    # Sound sources
    sources: List[SoundSource]
    
    # Ambient conditions
    ambient: Dict[str, AmbientCondition]
    
    # Background noise
    background_noise_level: float = 45.0  # dB SPL
    
    # Array geometry (ReSpeaker coordinates in meters)
    mic_positions: List[MicrophonePosition] = None
    
    def __post_init__(self):
        if self.mic_positions is None:
            # ReSpeaker USB 4 Mic Array geometry (exact coordinates from config)
            self.mic_positions = [
                MicrophonePosition(-0.032, 0.000, 0.000),  # Mic 2
                MicrophonePosition(0.000, -0.032, 0.000),  # Mic 3  
                MicrophonePosition(0.032, 0.000, 0.000),   # Mic 4
                MicrophonePosition(0.000, 0.032, 0.000)    # Mic 5
            ]

class AcousticPropagation:
    """Handles acoustic wave propagation physics"""
    
    def __init__(self, speed_of_sound: float = 343.0):
        self.c = speed_of_sound
    
    def calculate_distance(self, source_pos: Tuple[float, float, float], 
                          mic_pos: MicrophonePosition) -> float:
        """Calculate 3D distance between source and microphone"""
        return np.sqrt(
            (source_pos[0] - mic_pos.x)**2 + 
            (source_pos[1] - mic_pos.y)**2 + 
            (source_pos[2] - mic_pos.z)**2
        )
    
    def calculate_time_delay(self, source_pos: Tuple[float, float, float], 
                           mic_pos: MicrophonePosition) -> float:
        """Calculate propagation delay in seconds"""
        distance = self.calculate_distance(source_pos, mic_pos)
        return distance / self.c
    
    def calculate_attenuation(self, distance: float, reference_distance: float = 1.0) -> float:
        """Calculate amplitude attenuation due to distance (inverse square law)"""
        if distance <= 0:
            return 1.0
        return reference_distance / distance
    
    def apply_frequency_attenuation(self, audio: np.ndarray, distance: float, 
                                  sample_rate: int) -> np.ndarray:
        """Apply frequency-dependent atmospheric absorption"""
        if distance < 10.0:  # Skip for close sources
            return audio
            
        # Simple high-frequency rolloff for distant sources
        # Real implementation would use detailed atmospheric absorption models
        nyquist = sample_rate // 2
        cutoff = max(1000, nyquist - distance * 10)  # Hz
        
        if cutoff < nyquist:
            sos = signal.butter(3, cutoff, btype='low', fs=sample_rate, output='sos')
            return signal.sosfilt(sos, audio)
        
        return audio

class SoundLibrary:
    """Manages loading and preprocessing of audio files"""
    
    def __init__(self):
        self.loaded_sounds = {}
    
    def load_audio(self, file_path: str, target_sr: int = 16000) -> np.ndarray:
        """Load audio file with caching"""
        cache_key = f"{file_path}_{target_sr}"
        
        if cache_key not in self.loaded_sounds:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Audio file not found: {file_path}")
            
            audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
            self.loaded_sounds[cache_key] = audio
        
        return self.loaded_sounds[cache_key].copy()

class AmbientGenerator:
    """Generates ambient environmental sounds"""
    
    def __init__(self, sample_rate: int = 16000):
        self.sr = sample_rate
    
    def generate_rain(self, duration: float, intensity: str = 'moderate') -> np.ndarray:
        """Generate rain sound using filtered white noise"""
        num_samples = int(duration * self.sr)
        
        # Generate white noise
        noise = np.random.normal(0, 1, num_samples)
        
        # Apply filtering to create rain-like sound
        intensity_params = {
            'light': {'gain': 0.3, 'cutoff': 8000},
            'moderate': {'gain': 0.5, 'cutoff': 6000}, 
            'heavy': {'gain': 0.8, 'cutoff': 4000}
        }
        
        params = intensity_params.get(intensity, intensity_params['moderate'])
        
        # Band-pass filter to simulate rain frequency content
        sos_high = signal.butter(2, 200, btype='high', fs=self.sr, output='sos')
        sos_low = signal.butter(2, params['cutoff'], btype='low', fs=self.sr, output='sos')
        
        filtered = signal.sosfilt(sos_high, noise)
        filtered = signal.sosfilt(sos_low, filtered)
        
        return filtered * params['gain']
    
    def generate_wind(self, duration: float, speed_kmh: float = 15) -> np.ndarray:
        """Generate wind sound using colored noise"""
        num_samples = int(duration * self.sr)
        
        # Generate pink noise (1/f noise)
        white = np.random.normal(0, 1, num_samples)
        
        # Create 1/f characteristics
        freqs = np.fft.rfftfreq(num_samples, 1/self.sr)
        freqs[0] = 1  # Avoid division by zero
        
        # Pink noise filter (1/sqrt(f))
        pink_filter = 1 / np.sqrt(freqs)
        
        spectrum = np.fft.rfft(white)
        pink_spectrum = spectrum * pink_filter
        pink_noise = np.fft.irfft(pink_spectrum, n=num_samples)
        
        # Scale based on wind speed
        wind_scale = min(speed_kmh / 30.0, 1.0)  # Normalize to max 30 km/h
        
        # Low-pass filter for wind characteristics
        sos = signal.butter(3, 500, btype='low', fs=self.sr, output='sos')
        wind = signal.sosfilt(sos, pink_noise)
        
        return wind * wind_scale * 0.4
    
    def generate_forest_base(self, duration: float) -> np.ndarray:
        """Generate subtle forest ambient sound"""
        num_samples = int(duration * self.sr)
        
        # Very low-level broadband noise
        ambient = np.random.normal(0, 0.1, num_samples)
        
        # Add subtle low-frequency content
        sos = signal.butter(2, 300, btype='low', fs=self.sr, output='sos')
        ambient = signal.sosfilt(sos, ambient)
        
        return ambient * 0.2

class AcousticSceneGenerator:
    """Main class for generating synthetic acoustic scenes"""
    
    def __init__(self):
        self.sound_library = SoundLibrary()
        self.propagation = AcousticPropagation()
        self.ambient_gen = AmbientGenerator()
    
    def generate_scene(self, config: SimulationConfig) -> Tuple[np.ndarray, Dict]:
        """
        Generate complete acoustic scene
        
        Returns:
            mic_signals: (4, N) array of microphone signals
            metadata: Dictionary containing ground truth and simulation details
        """
        num_samples = int(config.scene_duration * config.sample_rate)
        num_mics = len(config.mic_positions)
        
        # Initialize output arrays
        mic_signals = np.zeros((num_mics, num_samples))
        
        # Generate ambient background
        ambient_signals = self._generate_ambient_background(config)
        mic_signals += ambient_signals
        
        # Add directional sound sources
        source_metadata = []
        for source in config.sources:
            source_signals, source_meta = self._add_sound_source(
                source, config, num_samples
            )
            mic_signals += source_signals
            source_metadata.append(source_meta)
        
        # Add background noise
        noise_signals = self._add_background_noise(config, num_samples)
        mic_signals += noise_signals
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(mic_signals))
        if max_val > 0.95:
            mic_signals = mic_signals * (0.95 / max_val)
        
        # Create comprehensive metadata
        metadata = {
            'config': config.__dict__,
            'sources': source_metadata,
            'scene_duration': config.scene_duration,
            'sample_rate': config.sample_rate,
            'mic_positions': [
                {'x': mic.x, 'y': mic.y, 'z': mic.z} 
                for mic in config.mic_positions
            ],
            'ground_truth': self._compute_ground_truth(config)
        }
        
        return mic_signals, metadata
    
    def _generate_ambient_background(self, config: SimulationConfig) -> np.ndarray:
        """Generate ambient environmental sounds"""
        num_mics = len(config.mic_positions)
        num_samples = int(config.scene_duration * config.sample_rate)
        ambient_signals = np.zeros((num_mics, num_samples))
        
        for condition_type, condition in config.ambient.items():
            if condition_type == 'rain':
                rain = self.ambient_gen.generate_rain(
                    config.scene_duration, condition.intensity
                )
                # Rain is omnidirectional - same for all mics with slight variations
                for i in range(num_mics):
                    variation = np.random.normal(1.0, 0.05)  # 5% variation
                    ambient_signals[i] += rain * variation * (10**(condition.level_db/20))
            
            elif condition_type == 'wind':
                wind = self.ambient_gen.generate_wind(
                    config.scene_duration, condition.intensity
                )
                for i in range(num_mics):
                    variation = np.random.normal(1.0, 0.1)  # 10% variation for wind
                    ambient_signals[i] += wind * variation * (10**(condition.level_db/20))
            
            elif condition_type == 'forest_base':
                forest = self.ambient_gen.generate_forest_base(config.scene_duration)
                for i in range(num_mics):
                    variation = np.random.normal(1.0, 0.03)  # 3% variation
                    ambient_signals[i] += forest * variation * (10**(condition.level_db/20))
        
        return ambient_signals
    
    def _add_sound_source(self, source: SoundSource, config: SimulationConfig,
                         num_samples: int) -> Tuple[np.ndarray, Dict]:
        """Add a directional sound source to the scene"""
        num_mics = len(config.mic_positions)
        source_signals = np.zeros((num_mics, num_samples))
        
        # Load source audio
        audio = self.sound_library.load_audio(source.audio_file, config.sample_rate)
        audio *= source.volume_scale
        
        # Calculate timing and placement
        start_sample = int(source.start_time * config.sample_rate)
        source_length = len(audio)
        
        # Trim if source extends beyond scene duration
        if start_sample + source_length > num_samples:
            source_length = num_samples - start_sample
            audio = audio[:source_length]
        
        if start_sample >= num_samples or source_length <= 0:
            # Source is outside scene duration
            return source_signals, self._create_source_metadata(source, config, [])
        
        # Calculate propagation to each microphone
        mic_delays = []
        mic_attenuations = []
        processed_audio_per_mic = []
        
        for i, mic_pos in enumerate(config.mic_positions):
            # Calculate propagation delay and attenuation
            delay = self.propagation.calculate_time_delay(source.position, mic_pos)
            distance = self.propagation.calculate_distance(source.position, mic_pos)
            attenuation = self.propagation.calculate_attenuation(distance)
            
            # Apply frequency-dependent attenuation for distant sources
            processed_audio = self.propagation.apply_frequency_attenuation(
                audio, distance, config.sample_rate
            )
            
            # Apply amplitude attenuation
            processed_audio *= attenuation
            
            # Calculate delay in samples
            delay_samples = int(delay * config.sample_rate)
            
            # Place audio in timeline with proper delay
            if start_sample + delay_samples >= 0 and start_sample + delay_samples < num_samples:
                end_sample = min(start_sample + delay_samples + len(processed_audio), num_samples)
                audio_end = end_sample - (start_sample + delay_samples)
                
                source_signals[i, start_sample + delay_samples:end_sample] += processed_audio[:audio_end]
            
            mic_delays.append(delay)
            mic_attenuations.append(attenuation)
            processed_audio_per_mic.append(processed_audio)
        
        metadata = self._create_source_metadata(source, config, mic_delays, mic_attenuations)
        
        return source_signals, metadata
    
    def _add_background_noise(self, config: SimulationConfig, num_samples: int) -> np.ndarray:
        """Add uncorrelated background noise to each microphone"""
        num_mics = len(config.mic_positions)
        noise_signals = np.zeros((num_mics, num_samples))
        
        # Convert dB to linear scale (assuming 0 dB = 1.0)
        noise_amplitude = 10**(config.background_noise_level / 20) * 0.01  # Scale factor
        
        for i in range(num_mics):
            noise = np.random.normal(0, noise_amplitude, num_samples)
            noise_signals[i] = noise
        
        return noise_signals
    
    def _create_source_metadata(self, source: SoundSource, config: SimulationConfig,
                               mic_delays: List[float], mic_attenuations: List[float] = None) -> Dict:
        """Create metadata for a sound source"""
        if mic_attenuations is None:
            mic_attenuations = []
            
        return {
            'audio_file': source.audio_file,
            'position': source.position,
            'start_time': source.start_time,
            'volume_scale': source.volume_scale,
            'source_type': source.source_type,
            'mic_delays': mic_delays,
            'mic_attenuations': mic_attenuations,
            'distances': [
                self.propagation.calculate_distance(source.position, mic_pos)
                for mic_pos in config.mic_positions
            ]
        }
    
    def _compute_ground_truth(self, config: SimulationConfig) -> Dict:
        """Compute ground truth DOA and other metrics"""
        ground_truth = {
            'source_directions': [],
            'expected_tdoas': [],
            'source_ranges': []
        }
        
        for source in config.sources:
            # Compute azimuth angle from array center
            azimuth = np.arctan2(source.position[1], source.position[0]) * 180 / np.pi
            if azimuth < 0:
                azimuth += 360
            
            # Compute range
            range_from_center = np.sqrt(source.position[0]**2 + source.position[1]**2)
            
            # Compute expected TDOAs between mic pairs
            tdoas = []
            for i in range(len(config.mic_positions)):
                for j in range(i+1, len(config.mic_positions)):
                    delay_i = self.propagation.calculate_time_delay(source.position, config.mic_positions[i])
                    delay_j = self.propagation.calculate_time_delay(source.position, config.mic_positions[j])
                    tdoa = delay_j - delay_i
                    tdoas.append({
                        'mic_pair': (i, j),
                        'tdoa_seconds': tdoa,
                        'tdoa_samples': tdoa * config.sample_rate
                    })
            
            ground_truth['source_directions'].append({
                'azimuth_deg': azimuth,
                'elevation_deg': 0.0,  # Assuming ground-level sources
                'position': source.position
            })
            
            ground_truth['expected_tdoas'].append(tdoas)
            ground_truth['source_ranges'].append(range_from_center)
        
        return ground_truth

# Example usage and utility functions
def create_example_config() -> SimulationConfig:
    """Create an example simulation configuration"""
    
    # Define sound sources
    sources = [
        SoundSource(
            audio_file="elephant_call.wav",
            position=(100.0, 0.0, 0.0),  # 100m North
            start_time=2.0,
            volume_scale=1.0,
            source_type="elephant"
        ),
        SoundSource(
            audio_file="bird_call.wav", 
            position=(50.0, 50.0, 5.0),  # 50m NE, 5m elevation
            start_time=0.5,
            volume_scale=0.7,
            source_type="bird"
        ),
        SoundSource(
            audio_file="vehicle.wav",
            position=(0.0, -200.0, 0.0),  # 200m South
            start_time=4.0,
            volume_scale=0.9,
            source_type="vehicle"
        )
    ]
    
    # Define ambient conditions
    ambient = {
        'rain': AmbientCondition('rain', 'light', 35.0),
        'wind': AmbientCondition('wind', 12.0, 28.0),  # 12 km/h
        'forest_base': AmbientCondition('forest_base', 'low', 25.0)
    }
    
    return SimulationConfig(
        scene_duration=10.0,
        sample_rate=16000,
        sources=sources,
        ambient=ambient,
        background_noise_level=45.0
    )

def save_simulation_results(mic_signals: np.ndarray, metadata: Dict, 
                          output_dir: str, scene_id: str = "scene_001"):
    """Save simulation results to files"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save 4-channel audio (each channel as separate file for compatibility)
    for i, channel in enumerate(mic_signals):
        np.save(output_path / f"{scene_id}_mic_{i}.npy", channel)
    
    # Save as multichannel numpy array
    np.save(output_path / f"{scene_id}_all_channels.npy", mic_signals)
    
    # Save metadata as JSON
    with open(output_path / f"{scene_id}_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    print(f"Simulation results saved to {output_path}")
    print(f"- Audio files: {scene_id}_mic_[0-3].npy")
    print(f"- Full array: {scene_id}_all_channels.npy") 
    print(f"- Metadata: {scene_id}_metadata.json")

if __name__ == "__main__":
    # Example usage
    generator = AcousticSceneGenerator()
    
    # Create example configuration
    config = create_example_config()
    
    # Generate scene (this will fail without actual audio files)
    try:
        mic_signals, metadata = generator.generate_scene(config)
        save_simulation_results(mic_signals, metadata, "simulation_output")
        
        print(f"Generated scene with {len(config.sources)} sources")
        print(f"Scene duration: {config.scene_duration}s")
        print(f"Sample rate: {config.sample_rate} Hz")
        print(f"Output shape: {mic_signals.shape}")
        
    except FileNotFoundError as e:
        print(f"Audio file not found: {e}")
        print("Please provide actual audio files to run the simulation.")
        
        # Show what the configuration looks like
        print("\nExample configuration:")
        print(f"Sources: {len(config.sources)}")
        for i, source in enumerate(config.sources):
            print(f"  {i+1}. {source.source_type} at {source.position}")
        
        print(f"\nAmbient conditions: {list(config.ambient.keys())}")
        print(f"Background noise: {config.background_noise_level} dB")
