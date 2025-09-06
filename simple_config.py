"""
Simplified Scene Configuration for Wildlife Acoustic Simulation
Uses real audio data from audio_cache instead of hardcoded sounds
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np
from pathlib import Path
import json

@dataclass
class SoundPlacement:
    """Simple sound placement in 3D space"""
    audio_file: str      # Path to real audio file from audio_cache
    position: Tuple[float, float, float]  # (x, y, z) in meters, mic array at origin
    start_time: float    # When sound starts in scene (seconds)
    volume: float = 1.0  # Volume scaling factor (0.0 to 2.0)
    sound_type: str = "unknown"  # bird, elephant, ambient, etc.
    distance: float = 0.0      # Distance from origin (calculated automatically)
    azimuth: float = 0.0       # Azimuth angle in degrees (0° = North, 90° = East)
    elevation: float = 0.0     # Elevation angle in degrees (0° = horizontal)
    
    def __post_init__(self):
        """Calculate distance from position"""
        self.distance = np.sqrt(self.position[0]**2 + self.position[1]**2 + self.position[2]**2)
        
        # Calculate azimuth (0° = North/+Y, 90° = East/+X)
        self.azimuth = np.degrees(np.arctan2(self.position[0], self.position[1]))
        if self.azimuth < 0:
            self.azimuth += 360
            
        # Calculate elevation
        xy_dist = np.sqrt(self.position[0]**2 + self.position[1]**2)
        self.elevation = np.degrees(np.arctan2(self.position[2], xy_dist))

@dataclass
class AmbientSound:
    """Ambient/background sounds"""
    sound_type: str      # rain, wind, forest, etc.
    level_db: float      # Background level in dB
    audio_file: Optional[str] = None  # Optional specific ambient file

@dataclass
class SceneConfig:
    """Simple scene configuration"""
    name: str
    duration: float      # Scene duration in seconds
    sounds: List[SoundPlacement]
    ambient: List[AmbientSound]
    sample_rate: int = 16000
    description: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'name': self.name,
            'duration': self.duration,
            'sounds': [
                {
                    'audio_file': s.audio_file,
                    'position': s.position,
                    'distance': s.distance,
                    'azimuth': s.azimuth,
                    'elevation': s.elevation,
                    'start_time': s.start_time,
                    'volume': s.volume,
                    'sound_type': s.sound_type
                }
                for s in self.sounds
            ],
            'ambient': [
                {
                    'sound_type': a.sound_type,
                    'level_db': a.level_db,
                    'audio_file': a.audio_file
                }
                for a in self.ambient
            ],
            'sample_rate': self.sample_rate,
            'description': self.description
        }
    
    def save(self, filepath: str):
        """Save configuration to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'SceneConfig':
        """Load configuration from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        sounds = [
            SoundPlacement(
                audio_file=s['audio_file'],
                position=tuple(s['position']),
                start_time=s['start_time'],
                volume=s.get('volume', 1.0),
                sound_type=s.get('sound_type', 'unknown')
            )
            for s in data['sounds']
        ]
        
        ambient = [
            AmbientSound(
                sound_type=a['sound_type'],
                level_db=a['level_db'],
                audio_file=a.get('audio_file')
            )
            for a in data['ambient']
        ]
        
        return cls(
            name=data['name'],
            duration=data['duration'],
            sounds=sounds,
            ambient=ambient,
            sample_rate=data.get('sample_rate', 16000),
            description=data.get('description', '')
        )

class SceneBuilder:
    """Helper to build scenes using audio_cache data"""
    
    def __init__(self, dataset_manager):
        self.dataset_manager = dataset_manager
        self.catalog = dataset_manager.catalog
    
    def find_audio_by_type(self, sound_type: str, dataset: str = None) -> List[str]:
        """Find audio files by type from catalog"""
        matches = []
        
        for file_hash, file_info in self.catalog.catalog["files"].items():
            abs_path = file_info["absolute_path"]
            
            # Filter by dataset if specified
            if dataset and file_info.get("dataset") != dataset:
                continue
            
            # Simple type matching based on path/filename
            path_lower = abs_path.lower()
            if sound_type.lower() in path_lower:
                matches.append(abs_path)
        
        return matches
    
    def select_audio_for_type(self, sound_type: str, dataset: str = None) -> str:
        """Select a random audio file matching the sound type"""
        import random
        matches = self.find_audio_by_type(sound_type, dataset)
        return random.choice(matches) if matches else None
    
    def place_sound_at_angle(self, audio_file: str, distance: float, azimuth_deg: float, 
                           elevation_deg: float = 0.0, start_time: float = 0.0, 
                           volume: float = 1.0, sound_type: str = "unknown") -> SoundPlacement:
        """Create a sound placement at specific angle and distance"""
        
        # Convert spherical to cartesian (mic array at origin)
        # Azimuth: 0° = North (+Y), 90° = East (+X)
        azimuth_rad = np.radians(azimuth_deg)
        elevation_rad = np.radians(elevation_deg)
        
        x = distance * np.sin(azimuth_rad) * np.cos(elevation_rad)
        y = distance * np.cos(azimuth_rad) * np.cos(elevation_rad) 
        z = distance * np.sin(elevation_rad)
        
        return SoundPlacement(
            audio_file=audio_file,
            position=(x, y, z),
            start_time=start_time,
            volume=volume,
            sound_type=sound_type
        )
    
    def create_simple_scene(self, name: str, duration: float = 10.0) -> SceneConfig:
        """Create a simple scene template"""
        return SceneConfig(
            name=name,
            duration=duration,
            sounds=[],
            ambient=[
                AmbientSound("forest", -45.0)  # Basic forest background
            ],
            description=f"Simple scene: {name}"
        )

# Example usage and presets
def create_elephant_scene_example(scene_builder: SceneBuilder) -> SceneConfig:
    """Create example elephant detection scene using real audio"""
    
    # Find elephant sounds
    elephant_files = scene_builder.find_audio_by_type("elephant")
    bird_files = scene_builder.find_audio_by_type("bird") 
    
    if not elephant_files:
        elephant_files = scene_builder.find_audio_by_type("animal")  # Fallback
    
    scene = scene_builder.create_simple_scene("Elephant Detection", 15.0)
    
    # Add elephant at 120m, 45° (Northeast)
    if elephant_files:
        scene.sounds.append(
            scene_builder.place_sound_at_angle(
                audio_file=elephant_files[0],
                distance=120.0,
                azimuth_deg=45.0,
                start_time=2.0,
                volume=1.0,
                sound_type="elephant"
            )
        )
    
    # Add bird distractor at 30m, 180° (South)  
    if bird_files:
        scene.sounds.append(
            scene_builder.place_sound_at_angle(
                audio_file=bird_files[0],
                distance=30.0,
                azimuth_deg=180.0,
                start_time=1.0,
                volume=0.6,
                sound_type="bird"
            )
        )
    
    return scene