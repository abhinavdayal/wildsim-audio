"""
Enhanced Pydantic Models for Wildlife Acoustic Scene Simulation
Provides robust validation, configuration management, and type safety
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Dict, Tuple, Optional, Union, Literal
from pathlib import Path
import numpy as np
from enum import Enum
import os

class SourceType(str, Enum):
    """Supported sound source types"""
    # Original categories
    ELEPHANT = "elephant"
    BIRD = "bird"
    VEHICLE = "vehicle"
    MONKEY = "monkey"
    HUMAN_ACTIVITY = "human_activity"
    MACHINERY = "machinery"
    WATER = "water"
    INSECT = "insect"
    UNKNOWN = "unknown"
    
    # Forest dataset sound classes
    AXE = "axe"
    BIRDCHIRPING = "birdchirping"
    CHAINSAW = "chainsaw"
    CLAPPING = "clapping"
    FIRE = "fire"
    FIREWORK = "firework"
    FOOTSTEPS = "footsteps"
    FROG = "frog"
    GENERATOR = "generator"
    GUNSHOT = "gunshot"
    HANDSAW = "handsaw"
    HELICOPTER = "helicopter"
    LION = "lion"
    RAIN = "rain"
    SILENCE = "silence"
    SPEAKING = "speaking"
    SQUIRREL = "squirrel"
    THUNDERSTORM = "thunderstorm"
    TREEFALLING = "treefalling"
    VEHICLEENGINE = "vehicleengine"
    WATERDROPS = "waterdrops"
    WHISTLING = "whistling"
    WIND = "wind"
    WINGFLAPING = "wingflaping"
    WOLFHOWL = "wolfhowl"
    WOODCHOP = "woodchop"
    
    # Wild animals dataset classes
    BEAR = "bear"
    CAT = "cat"
    CHICKEN = "chicken"
    COW = "cow"
    DOG = "dog"
    DOLPHIN = "dolphin"
    DONKEY = "donkey"
    HORSE = "horse"
    SHEEP = "sheep"

class AmbientType(str, Enum):
    """Supported ambient condition types"""
    RAIN = "rain"
    WIND = "wind"
    FOREST_BASE = "forest_base"
    WATER_SOURCE = "water_source"

class RainIntensity(str, Enum):
    """Rain intensity levels"""
    LIGHT = "light"
    MODERATE = "moderate"
    HEAVY = "heavy"

class EnvironmentType(str, Enum):
    """Predefined environment types"""
    FOREST_CLEARING = "forest_clearing"
    DENSE_FOREST = "dense_forest"
    SAVANNA = "savanna"
    CUSTOM = "custom"

class MicrophonePosition(BaseModel):
    """3D microphone position with validation"""
    x: float = Field(..., description="X coordinate in meters")
    y: float = Field(..., description="Y coordinate in meters") 
    z: float = Field(0.0, description="Z coordinate in meters (height)")
    
    @field_validator('z')
    @classmethod
    def validate_height(cls, v):
        if v < -10 or v > 100:
            raise ValueError("Microphone height must be between -10m and 100m")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "x": -0.032,
                "y": 0.000,
                "z": 0.000
            }
        }

class SoundSource(BaseModel):
    """Sound source configuration with comprehensive validation"""
    
    audio_file: str = Field(..., description="Path to audio file")
    position: Tuple[float, float, float] = Field(..., description="3D position (x, y, z) in meters")
    start_time: float = Field(0.0, ge=0.0, description="Start time in seconds")
    volume_scale: float = Field(1.0, gt=0.0, le=10.0, description="Volume scaling factor")
    source_type: SourceType = Field(SourceType.UNKNOWN, description="Type of sound source")
    duration_override: Optional[float] = Field(None, gt=0.0, le=300.0, description="Override audio duration (seconds)")
    
    @field_validator('position')
    @classmethod
    def validate_position(cls, v):
        x, y, z = v
        
        # Check maximum distance (10km)
        distance_2d = (x**2 + y**2)**0.5
        if distance_2d > 10000:
            raise ValueError(f"Source distance {distance_2d:.0f}m exceeds maximum 10km")
        
        # Check height constraints
        if z < -50 or z > 1000:
            raise ValueError(f"Source height {z}m must be between -50m and 1000m")
        
        return v
    
    @field_validator('audio_file')
    @classmethod
    def validate_audio_file_extension(cls, v):
        valid_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg'}
        if not any(v.lower().endswith(ext) for ext in valid_extensions):
            raise ValueError(f"Audio file must have extension: {', '.join(valid_extensions)}")
        return v
    
    @property
    def distance_from_origin(self) -> float:
        """Calculate 2D distance from coordinate origin"""
        return (self.position[0]**2 + self.position[1]**2)**0.5
    
    @property
    def azimuth_deg(self) -> float:
        """Calculate azimuth in degrees"""
        azimuth = np.arctan2(self.position[1], self.position[0]) * 180 / np.pi
        return azimuth if azimuth >= 0 else azimuth + 360
    
    class Config:
        use_enum_values = True
        json_schema_extra = {
            "example": {
                "audio_file": "sounds/elephant_call.wav",
                "position": [100.0, 0.0, 0.0],
                "start_time": 2.0,
                "volume_scale": 1.0,
                "source_type": "elephant",
                "duration_override": None
            }
        }

class AmbientCondition(BaseModel):
    """Ambient environmental condition with validation"""
    
    condition_type: AmbientType = Field(..., description="Type of ambient condition")
    intensity: Union[RainIntensity, float] = Field(..., description="Intensity (string for rain, float for wind speed)")
    level_db: float = Field(..., ge=0.0, le=80.0, description="Sound level in dB")
    
    @model_validator(mode='before')
    @classmethod
    def validate_intensity_type(cls, values):
        if isinstance(values, dict):
            condition_type = values.get('condition_type')
            intensity = values.get('intensity')
            
            if condition_type == AmbientType.RAIN:
                if not isinstance(intensity, str) or intensity not in [e.value for e in RainIntensity]:
                    raise ValueError(f"Rain intensity must be one of: {[e.value for e in RainIntensity]}")
            elif condition_type == AmbientType.WIND:
                if not isinstance(intensity, (int, float)) or intensity < 0 or intensity > 200:
                    raise ValueError("Wind speed must be between 0 and 200 km/h")
        
        return values
    
    class Config:
        use_enum_values = True

class EnvironmentConfig(BaseModel):
    """Acoustic environment configuration with physics-based validation"""
    
    environment_type: EnvironmentType = Field(EnvironmentType.FOREST_CLEARING, description="Predefined environment type")
    room_size: Tuple[float, float, float] = Field((1000.0, 1000.0, 50.0), description="Environment size (x, y, z) in meters")
    absorption_coefficient: float = Field(0.02, ge=0.001, le=0.9, description="Acoustic absorption coefficient")
    max_order: int = Field(2, ge=0, le=10, description="Maximum reflection order")
    temperature: float = Field(20.0, ge=-40.0, le=60.0, description="Temperature in Celsius")
    humidity: float = Field(50.0, ge=0.0, le=100.0, description="Relative humidity percentage")
    air_absorption: bool = Field(True, description="Include frequency-dependent air absorption")
    
    @field_validator('room_size')
    @classmethod
    def validate_room_size(cls, v):
        x, y, z = v
        if x <= 0 or y <= 0 or z <= 0:
            raise ValueError("Room dimensions must be positive")
        if x > 50000 or y > 50000 or z > 5000:
            raise ValueError("Room size unrealistically large (max: 50km x 50km x 5km)")
        return v
    
    @field_validator('absorption_coefficient')
    @classmethod
    def validate_absorption(cls, v):
        if v < 0.001:
            raise ValueError("Absorption coefficient too low (min: 0.001 for outdoor)")
        if v > 0.5:
            raise ValueError("Absorption coefficient too high (max: 0.5 for very absorptive)")
        return v
    
    class Config:
        use_enum_values = True
        json_schema_extra = {
            "example": {
                "environment_type": "forest_clearing",
                "room_size": [1000.0, 1000.0, 50.0],
                "absorption_coefficient": 0.02,
                "max_order": 2,
                "temperature": 25.0,
                "humidity": 60.0,
                "air_absorption": True
            }
        }

class ProcessingConfig(BaseModel):
    """Signal processing parameters with validation"""
    
    sample_rate: int = Field(16000, description="Audio sample rate in Hz")
    frame_size: int = Field(512, description="STFT frame size in samples")
    hop_length: int = Field(256, description="STFT hop length in samples")
    window_type: Literal["hann", "hamming", "blackman"] = Field("hann", description="Window function type")
    
    @field_validator('sample_rate')
    @classmethod
    def validate_sample_rate(cls, v):
        valid_rates = [8000, 16000, 22050, 44100, 48000, 96000]
        if v not in valid_rates:
            raise ValueError(f"Sample rate must be one of: {valid_rates}")
        return v
    
    @field_validator('frame_size')
    @classmethod
    def validate_frame_size(cls, v):
        # Must be power of 2 for efficient FFT
        if v <= 0 or (v & (v-1)) != 0:
            raise ValueError("Frame size must be a positive power of 2")
        if v < 64 or v > 8192:
            raise ValueError("Frame size must be between 64 and 8192")
        return v
    
    @model_validator(mode='after')
    def validate_hop_length(self):
        if self.hop_length <= 0 or self.hop_length > self.frame_size:
            raise ValueError("Hop length must be positive and <= frame_size")
        return self
    
    class Config:
        json_schema_extra = {
            "example": {
                "sample_rate": 16000,
                "frame_size": 512,
                "hop_length": 256,
                "window_type": "hann"
            }
        }

class DatasetConfig(BaseModel):
    """Audio dataset configuration"""
    
    dataset_name: str = Field(..., description="Name of the dataset")
    source_type: Literal["local", "kaggle", "huggingface", "google_drive", "url"] = Field("local", description="Dataset source")
    path_or_id: str = Field(..., description="Local path, Kaggle dataset ID, or URL")
    audio_extensions: List[str] = Field([".wav", ".mp3", ".flac"], description="Supported audio file extensions")
    metadata_file: Optional[str] = Field(None, description="Path to metadata CSV/JSON file")
    cache_dir: Optional[str] = Field("./audio_cache", description="Local cache directory")
    
    @field_validator('audio_extensions')
    @classmethod
    def validate_extensions(cls, v):
        valid_exts = {'.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac'}
        for ext in v:
            if ext.lower() not in valid_exts:
                raise ValueError(f"Unsupported audio extension: {ext}")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "dataset_name": "ESC-50",
                "source_type": "kaggle",
                "path_or_id": "karoldvl/esc50-dataset",
                "audio_extensions": [".wav"],
                "metadata_file": "meta/esc50.csv",
                "cache_dir": "./audio_cache/esc50"
            }
        }

class SimulationConfig(BaseModel):
    """Complete simulation configuration with comprehensive validation"""
    
    # Basic parameters
    scene_duration: float = Field(10.0, gt=0.0, le=300.0, description="Scene duration in seconds")
    background_noise_level: float = Field(45.0, ge=0.0, le=120.0, description="Background noise level in dB SPL")
    
    # Core components
    sources: List[SoundSource] = Field(..., min_items=1, description="Sound sources in the scene")
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig, description="Acoustic environment")
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig, description="Signal processing parameters")
    
    # Optional components
    ambient: Dict[str, AmbientCondition] = Field(default_factory=dict, description="Ambient environmental conditions")
    mic_positions: Optional[List[MicrophonePosition]] = Field(None, description="Microphone array geometry")
    datasets: List[DatasetConfig] = Field(default_factory=list, description="Audio datasets to use")
    
    # Metadata
    scene_name: str = Field("default_scene", description="Descriptive name for the scene")
    description: Optional[str] = Field(None, description="Detailed scene description")
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")
    
    def __init__(self, **data):
        super().__init__(**data)
        # Set default ReSpeaker microphone positions if not provided
        if self.mic_positions is None:
            self.mic_positions = [
                MicrophonePosition(x=-0.032, y=0.000, z=0.000),  # Mic 2
                MicrophonePosition(x=0.000, y=-0.032, z=0.000),  # Mic 3  
                MicrophonePosition(x=0.032, y=0.000, z=0.000),   # Mic 4
                MicrophonePosition(x=0.000, y=0.032, z=0.000)    # Mic 5
            ]
    
    @model_validator(mode='after')
    def validate_sources_fit_in_environment(self):
        if not self.environment:
            return self
            
        room_x, room_y, room_z = self.environment.room_size
        
        for i, source in enumerate(self.sources):
            x, y, z = source.position
            
            # Check if source is within environment bounds (with margin)
            margin = 10.0  # 10m margin from walls
            if not (-room_x/2 + margin <= x <= room_x/2 - margin):
                raise ValueError(f"Source {i+1} X position {x} outside environment bounds")
            if not (-room_y/2 + margin <= y <= room_y/2 - margin):
                raise ValueError(f"Source {i+1} Y position {y} outside environment bounds")
            if not (0 <= z <= room_z - margin):
                raise ValueError(f"Source {i+1} Z position {z} outside environment bounds")
        
        # Validate source start times
        for source in self.sources:
            if source.start_time >= self.scene_duration:
                raise ValueError(f"Source start time {source.start_time}s >= scene duration {self.scene_duration}s")
        
        return self
    
    @property
    def total_sources(self) -> int:
        """Total number of sources"""
        return len(self.sources)
    
    @property
    def elephant_sources(self) -> List[SoundSource]:
        """Filter sources by elephant type"""
        return [s for s in self.sources if s.source_type == SourceType.ELEPHANT]
    
    @property
    def source_summary(self) -> Dict[str, int]:
        """Summary of sources by type"""
        summary = {}
        for source in self.sources:
            source_type = source.source_type.value
            summary[source_type] = summary.get(source_type, 0) + 1
        return summary
    
    class Config:
        validate_assignment = True  # Validate when attributes are changed
        extra = "forbid"           # Don't allow unknown fields
        use_enum_values = True
        json_schema_extra = {
            "example": {
                "scene_duration": 12.0,
                "background_noise_level": 45.0,
                "sources": [
                    {
                        "audio_file": "sounds/elephant_call.wav",
                        "position": [120.0, 30.0, 0.0],
                        "start_time": 2.0,
                        "volume_scale": 1.0,
                        "source_type": "elephant"
                    }
                ],
                "environment": {
                    "environment_type": "forest_clearing",
                    "room_size": [1000.0, 1000.0, 50.0],
                    "absorption_coefficient": 0.02
                },
                "ambient": {
                    "wind": {
                        "condition_type": "wind",
                        "intensity": 15.0,
                        "level_db": 28.0
                    }
                },
                "scene_name": "elephant_detection_scenario",
                "tags": ["elephant", "validation"]
            }
        }

class ValidationResult(BaseModel):
    """Results from DOA algorithm validation"""
    
    method_name: str = Field(..., description="DOA algorithm name")
    azimuth_estimates: List[float] = Field(default_factory=list, description="Estimated azimuth angles in degrees")
    elevation_estimates: List[float] = Field(default_factory=list, description="Estimated elevation angles in degrees")
    confidence_scores: List[float] = Field(default_factory=list, description="Confidence scores for each estimate")
    spatial_spectrum: List[float] = Field(default_factory=list, description="Spatial spectrum values")
    processing_time: float = Field(0.0, description="Processing time in seconds")
    success: bool = Field(True, description="Whether the algorithm succeeded")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    
    class Config:
        json_schema_extra = {
            "example": {
                "method_name": "MUSIC",
                "azimuth_estimates": [45.2, 130.7],
                "elevation_estimates": [0.0, 5.3],
                "confidence_scores": [0.92, 0.78],
                "processing_time": 0.045,
                "success": True
            }
        }

class SimulationResult(BaseModel):
    """Complete simulation results with metadata"""
    
    config: SimulationConfig = Field(..., description="Original simulation configuration")
    mic_signals_shape: Tuple[int, int] = Field(..., description="Shape of microphone signals array (n_mics, n_samples)")
    peak_amplitude: float = Field(..., description="Peak signal amplitude")
    rms_level: float = Field(..., description="RMS signal level")
    dynamic_range_db: float = Field(..., description="Dynamic range in dB")
    
    # Ground truth
    ground_truth_azimuths: List[float] = Field(default_factory=list, description="True azimuth angles")
    ground_truth_distances: List[float] = Field(default_factory=list, description="True distances from array center")
    expected_tdoas: List[Dict] = Field(default_factory=list, description="Expected time differences of arrival")
    
    # Validation results
    validation_results: List[ValidationResult] = Field(default_factory=list, description="DOA algorithm validation results")
    
    # Processing metadata
    generation_time: float = Field(0.0, description="Time to generate scene in seconds")
    timestamp: str = Field(..., description="Timestamp when simulation was run")
    
    class Config:
        json_schema_extra = {
            "example": {
                "mic_signals_shape": [4, 160000],
                "peak_amplitude": 0.89,
                "rms_level": 0.12,
                "dynamic_range_db": 17.3,
                "ground_truth_azimuths": [45.0],
                "ground_truth_distances": [120.0],
                "generation_time": 2.34,
                "timestamp": "2024-01-15T10:30:45Z"
            }
        }

# Utility functions for model creation
def create_elephant_source(distance: float, azimuth_deg: float, 
                          audio_file: str = "elephant_call.wav",
                          elevation: float = 0.0, start_time: float = 2.0,
                          volume: float = 1.0) -> SoundSource:
    """Create a validated elephant source"""
    azimuth_rad = np.radians(azimuth_deg)
    x = distance * np.cos(azimuth_rad)
    y = distance * np.sin(azimuth_rad)
    
    return SoundSource(
        audio_file=audio_file,
        position=(x, y, elevation),
        start_time=start_time,
        volume_scale=volume,
        source_type=SourceType.ELEPHANT
    )

def create_forest_environment(size_km: float = 1.0, 
                             temperature: float = 25.0,
                             humidity: float = 60.0) -> EnvironmentConfig:
    """Create a validated forest environment"""
    size_m = size_km * 1000
    
    return EnvironmentConfig(
        environment_type=EnvironmentType.FOREST_CLEARING,
        room_size=(size_m, size_m, 50.0),
        absorption_coefficient=0.02,
        max_order=2,
        temperature=temperature,
        humidity=humidity,
        air_absorption=True
    )

def validate_config_file(config_path: str) -> bool:
    """Validate a configuration file without loading"""
    try:
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(config_path)
        SimulationConfig(**cfg)
        return True
    except Exception as e:
        print(f"Configuration validation failed: {e}")
        return False

if __name__ == "__main__":
    # Example usage and validation
    print("Testing Pydantic models...")
    
    # Test valid configuration
    try:
        config = SimulationConfig(
            scene_duration=10.0,
            sources=[
                create_elephant_source(100, 45, "sounds/elephant.wav")
            ],
            environment=create_forest_environment(),
            ambient={
                "wind": AmbientCondition(
                    condition_type=AmbientType.WIND,
                    intensity=15.0,
                    level_db=28.0
                )
            }
        )
        print("✓ Valid configuration created successfully")
        print(f"  Sources: {config.total_sources}")
        print(f"  Summary: {config.source_summary}")
        
    except Exception as e:
        print(f"✗ Configuration validation failed: {e}")
    
    # Test invalid configuration
    try:
        invalid_config = SimulationConfig(
            scene_duration=-5.0,  # Invalid: negative duration
            sources=[],           # Invalid: no sources
        )
        print("✗ Should have failed validation!")
        
    except Exception as e:
        print(f"✓ Correctly caught invalid configuration: {type(e).__name__}")
    
    print("Model validation tests completed!")