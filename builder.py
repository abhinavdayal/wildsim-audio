"""
Configuration Builder for Acoustic Scene Simulation
Provides easy-to-use interface for creating simulation scenarios
"""

from acoustic_scene_generator import *
import json
from typing import Dict, List
import argparse

class SceneConfigBuilder:
    """Helper class to build simulation configurations interactively"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset to default configuration"""
        self.duration = 10.0
        self.sample_rate = 16000
        self.sources = []
        self.ambient = {}
        self.noise_level = 45.0
    
    def add_elephant(self, distance: float, azimuth_deg: float, 
                    elevation: float = 0.0, start_time: float = 2.0,
                    audio_file: str = "elephant_call.wav", volume: float = 1.0):
        """Add elephant source at specified polar coordinates"""
        
        # Convert polar to cartesian
        azimuth_rad = np.radians(azimuth_deg)
        x = distance * np.cos(azimuth_rad)
        y = distance * np.sin(azimuth_rad)
        z = elevation
        
        source = SoundSource(
            audio_file=audio_file,
            position=(x, y, z),
            start_time=start_time,
            volume_scale=volume,
            source_type="elephant"
        )
        
        self.sources.append(source)
        return self
    
    def add_bird(self, distance: float, azimuth_deg: float,
                elevation: float = 5.0, start_time: float = 1.0,
                audio_file: str = "bird_call.wav", volume: float = 0.7):
        """Add bird source at specified location"""
        
        azimuth_rad = np.radians(azimuth_deg)
        x = distance * np.cos(azimuth_rad)
        y = distance * np.sin(azimuth_rad)
        z = elevation
        
        source = SoundSource(
            audio_file=audio_file,
            position=(x, y, z),
            start_time=start_time,
            volume_scale=volume,
            source_type="bird"
        )
        
        self.sources.append(source)
        return self
    
    def add_vehicle(self, distance: float, azimuth_deg: float,
                   start_time: float = 0.0, audio_file: str = "vehicle.wav",
                   volume: float = 0.8):
        """Add vehicle source (typically on roads)"""
        
        azimuth_rad = np.radians(azimuth_deg)
        x = distance * np.cos(azimuth_rad)
        y = distance * np.sin(azimuth_rad)
        z = 0.0  # Ground level
        
        source = SoundSource(
            audio_file=audio_file,
            position=(x, y, z),
            start_time=start_time,
            volume_scale=volume,
            source_type="vehicle"
        )
        
        self.sources.append(source)
        return self
    
    def add_custom_source(self, source_type: str, distance: float, 
                         azimuth_deg: float, elevation: float = 0.0,
                         start_time: float = 0.0, audio_file: str = "",
                         volume: float = 1.0):
        """Add custom source type"""
        
        azimuth_rad = np.radians(azimuth_deg)
        x = distance * np.cos(azimuth_rad)
        y = distance * np.sin(azimuth_rad)
        z = elevation
        
        if not audio_file:
            audio_file = f"{source_type}.wav"
        
        source = SoundSource(
            audio_file=audio_file,
            position=(x, y, z),
            start_time=start_time,
            volume_scale=volume,
            source_type=source_type
        )
        
        self.sources.append(source)
        return self
    
    def set_rain(self, intensity: str = "moderate", level_db: float = 35.0):
        """Add rain ambient condition"""
        self.ambient['rain'] = AmbientCondition('rain', intensity, level_db)
        return self
    
    def set_wind(self, speed_kmh: float = 15.0, level_db: float = 30.0):
        """Add wind ambient condition"""
        self.ambient['wind'] = AmbientCondition('wind', speed_kmh, level_db)
        return self
    
    def set_forest_ambient(self, level_db: float = 25.0):
        """Add forest base ambient"""
        self.ambient['forest_base'] = AmbientCondition('forest_base', 'medium', level_db)
        return self
    
    def set_duration(self, duration: float):
        """Set scene duration in seconds"""
        self.duration = duration
        return self
    
    def set_noise_level(self, level_db: float):
        """Set background noise level in dB"""
        self.noise_level = level_db
        return self
    
    def build(self) -> SimulationConfig:
        """Build the final configuration"""
        return SimulationConfig(
            scene_duration=self.duration,
            sample_rate=self.sample_rate,
            sources=self.sources.copy(),
            ambient=self.ambient.copy(),
            background_noise_level=self.noise_level
        )
    
    def preview(self):
        """Print a preview of the current configuration"""
        print("=== Scene Configuration Preview ===")
        print(f"Duration: {self.duration} seconds")
        print(f"Background noise: {self.noise_level} dB")
        print(f"Number of sources: {len(self.sources)}")
        
        print("\nSources:")
        for i, source in enumerate(self.sources):
            x, y, z = source.position
            distance = np.sqrt(x**2 + y**2)
            azimuth = np.degrees(np.arctan2(y, x))
            if azimuth < 0:
                azimuth += 360
                
            print(f"  {i+1}. {source.source_type}: {distance:.0f}m @ {azimuth:.0f}° "
                  f"(t={source.start_time:.1f}s, vol={source.volume_scale:.1f})")
        
        print(f"\nAmbient conditions: {list(self.ambient.keys())}")
        for condition_type, condition in self.ambient.items():
            print(f"  {condition_type}: {condition.intensity} ({condition.level_db:.0f} dB)")

def create_preset_scenarios():
    """Create dictionary of preset scenarios"""
    
    scenarios = {}
    
    # Simple elephant detection
    scenarios['simple_elephant'] = (
        SceneConfigBuilder()
        .add_elephant(distance=100, azimuth_deg=0)  # Due north
        .set_forest_ambient()
        .set_duration(8.0)
        .set_noise_level(40.0)
    )
    
    # Elephant with bird distractors
    scenarios['elephant_with_birds'] = (
        SceneConfigBuilder()
        .add_elephant(distance=150, azimuth_deg=45)
        .add_bird(distance=30, azimuth_deg=120, elevation=8.0)
        .add_bird(distance=50, azimuth_deg=300, elevation=12.0, start_time=4.0)
        .set_forest_ambient()
        .set_wind(speed_kmh=12)
        .set_duration(10.0)
    )
    
    # Multi-elephant herd
    scenarios['elephant_herd'] = (
        SceneConfigBuilder()
        .add_elephant(distance=200, azimuth_deg=0, start_time=1.0, 
                     audio_file="elephant_adult.wav")
        .add_elephant(distance=220, azimuth_deg=15, start_time=3.5,
                     audio_file="elephant_calf.wav", volume=0.7)
        .add_elephant(distance=180, azimuth_deg=-20, start_time=6.0,
                     audio_file="elephant_female.wav", volume=0.9)
        .set_forest_ambient(level_db=22.0)
        .set_duration(12.0)
        .set_noise_level(38.0)
    )
    
    # Noisy environment challenge
    scenarios['noisy_challenge'] = (
        SceneConfigBuilder()
        .add_elephant(distance=300, azimuth_deg=60, volume=0.8)  # Distant elephant
        .add_vehicle(distance=150, azimuth_deg=180, start_time=1.0, volume=1.2)
        .add_custom_source("machinery", distance=200, azimuth_deg=270, 
                          start_time=4.0, volume=1.1)
        .add_bird(distance=25, azimuth_deg=90, start_time=0.0, volume=0.8)
        .set_rain(intensity="moderate", level_db=38.0)
        .set_wind(speed_kmh=20, level_db=33.0)
        .set_noise_level(52.0)
        .set_duration(15.0)
    )
    
    # Validation scenario - clean single source
    scenarios['validation_clean'] = (
        SceneConfigBuilder()
        .add_elephant(distance=100, azimuth_deg=90, start_time=2.0)  # Due east
        .set_forest_ambient(level_db=20.0)
        .set_duration(6.0)
        .set_noise_level(30.0)
    )
    
    # Dawn chorus with elephant
    scenarios['dawn_chorus'] = (
        SceneConfigBuilder()
        .add_elephant(distance=250, azimuth_deg=30, start_time=5.0)
        .add_bird(distance=40, azimuth_deg=0, start_time=0.0, 
                 audio_file="hornbill.wav")
        .add_bird(distance=60, azimuth_deg=120, start_time=1.5,
                 audio_file="sunbird.wav", volume=0.6)
        .add_bird(distance=35, azimuth_deg=240, start_time=2.8,
                 audio_file="bulbul.wav", volume=0.8)
        .add_custom_source("monkey", distance=80, azimuth_deg=160,
                          start_time=7.0, elevation=15.0)
        .set_forest_ambient(level_db=28.0)
        .set_wind(speed_kmh=8, level_db=25.0)
        .set_duration(12.0)
        .set_noise_level(42.0)
    )
    
    return scenarios

def save_config_to_file(config: SimulationConfig, filename: str):
    """Save configuration to JSON file for later use"""
    
    # Convert to dictionary format
    config_dict = {
        'scene_duration': config.scene_duration,
        'sample_rate': config.sample_rate,
        'background_noise_level': config.background_noise_level,
        'sources': [
            {
                'audio_file': s.audio_file,
                'position': s.position,
                'start_time': s.start_time,
                'volume_scale': s.volume_scale,
                'source_type': s.source_type
            } for s in config.sources
        ],
        'ambient': {
            name: {
                'condition_type': cond.condition_type,
                'intensity': cond.intensity,
                'level_db': cond.level_db
            } for name, cond in config.ambient.items()
        },
        'mic_positions': [
            {'x': mic.x, 'y': mic.y, 'z': mic.z}
            for mic in config.mic_positions
        ]
    }
    
    with open(filename, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"Configuration saved to {filename}")

def load_config_from_file(filename: str) -> SimulationConfig:
    """Load configuration from JSON file"""
    
    with open(filename, 'r') as f:
        config_dict = json.load(f)
    
    # Reconstruct sources
    sources = [
        SoundSource(
            audio_file=s['audio_file'],
            position=tuple(s['position']),
            start_time=s['start_time'],
            volume_scale=s['volume_scale'],
            source_type=s['source_type']
        ) for s in config_dict['sources']
    ]
    
    # Reconstruct ambient conditions
    ambient = {
        name: AmbientCondition(
            condition_type=cond['condition_type'],
            intensity=cond['intensity'],
            level_db=cond['level_db']
        ) for name, cond in config_dict['ambient'].items()
    }
    
    # Reconstruct microphone positions
    mic_positions = [
        MicrophonePosition(mic['x'], mic['y'], mic['z'])
        for mic in config_dict['mic_positions']
    ]
    
    return SimulationConfig(
        scene_duration=config_dict['scene_duration'],
        sample_rate=config_dict['sample_rate'],
        sources=sources,
        ambient=ambient,
        background_noise_level=config_dict['background_noise_level'],
        mic_positions=mic_positions
    )

def interactive_config_builder():
    """Interactive command-line configuration builder"""
    
    builder = SceneConfigBuilder()
    
    print("=== Interactive Acoustic Scene Configuration ===")
    print("Commands: elephant, bird, vehicle, custom, rain, wind, forest, duration, noise, preview, done")
    
    while True:
        command = input("\nEnter command (or 'help' for options): ").strip().lower()
        
        if command == 'help':
            print("Available commands:")
            print("  elephant <distance> <azimuth> [elevation] [start_time] [volume]")
            print("  bird <distance> <azimuth> [elevation] [start_time] [volume]") 
            print("  vehicle <distance> <azimuth> [start_time] [volume]")
            print("  rain <intensity> [level_db] - intensity: light/moderate/heavy")
            print("  wind <speed_kmh> [level_db]")
            print("  forest [level_db] - forest ambient")
            print("  duration <seconds>")
            print("  noise <level_db> - background noise level")
            print("  preview - show current configuration")
            print("  done - finish and return configuration")
            
        elif command.startswith('elephant'):
            try:
                parts = command.split()[1:]
                distance = float(parts[0])
                azimuth = float(parts[1])
                elevation = float(parts[2]) if len(parts) > 2 else 0.0
                start_time = float(parts[3]) if len(parts) > 3 else 2.0
                volume = float(parts[4]) if len(parts) > 4 else 1.0
                
                builder.add_elephant(distance, azimuth, elevation, start_time, volume=volume)
                print(f"Added elephant at {distance}m, {azimuth}°")
            except (IndexError, ValueError):
                print("Usage: duration <seconds>")
        
        elif command.startswith('noise'):
            try:
                level = float(command.split()[1])
                builder.set_noise_level(level)
                print(f"Set background noise to {level} dB")
            except (IndexError, ValueError):
                print("Usage: noise <level_db>")
        
        elif command == 'forest':
            try:
                parts = command.split()[1:]
                level_db = float(parts[0]) if parts else 25.0
                builder.set_forest_ambient(level_db)
                print(f"Added forest ambient ({level_db} dB)")
            except ValueError:
                print("Usage: forest [level_db]")
        
        elif command == 'preview':
            builder.preview()
        
        elif command == 'done':
            if not builder.sources:
                print("Warning: No sources added to scene!")
                continue
            break
        
        else:
            print("Unknown command. Type 'help' for available commands.")
    
    return builder.build()

def main():
    """Main function for command-line usage"""
    
    parser = argparse.ArgumentParser(description="Acoustic Scene Configuration Builder")
    parser.add_argument("--preset", choices=list(create_preset_scenarios().keys()),
                       help="Use a preset scenario")
    parser.add_argument("--interactive", action="store_true",
                       help="Use interactive configuration builder")
    parser.add_argument("--save", type=str, help="Save configuration to file")
    parser.add_argument("--load", type=str, help="Load configuration from file")
    parser.add_argument("--preview", action="store_true", help="Preview configuration")
    parser.add_argument("--list-presets", action="store_true", 
                       help="List available preset scenarios")
    
    args = parser.parse_args()
    
    if args.list_presets:
        print("Available preset scenarios:")
        scenarios = create_preset_scenarios()
        for name, builder in scenarios.items():
            print(f"\n{name}:")
            builder.preview()
        return
    
    config = None
    
    if args.load:
        config = load_config_from_file(args.load)
        print(f"Loaded configuration from {args.load}")
    
    elif args.preset:
        scenarios = create_preset_scenarios()
        builder = scenarios[args.preset]
        config = builder.build()
        print(f"Using preset scenario: {args.preset}")
    
    elif args.interactive:
        config = interactive_config_builder()
    
    else:
        # Default: show preset options
        print("Available preset scenarios:")
        scenarios = create_preset_scenarios()
        names = list(scenarios.keys())
        
        for i, name in enumerate(names, 1):
            print(f"{i}. {name}")
        
        while True:
            try:
                choice = input(f"\nSelect scenario (1-{len(names)}) or 'i' for interactive: ")
                if choice.lower() == 'i':
                    config = interactive_config_builder()
                    break
                else:
                    idx = int(choice) - 1
                    if 0 <= idx < len(names):
                        selected = names[idx]
                        config = scenarios[selected].build()
                        print(f"Selected: {selected}")
                        break
                    else:
                        print("Invalid selection")
            except ValueError:
                print("Invalid input")
    
    if config and args.preview:
        # Create a temporary builder to use the preview method
        temp_builder = SceneConfigBuilder()
        temp_builder.sources = config.sources
        temp_builder.ambient = config.ambient
        temp_builder.duration = config.scene_duration
        temp_builder.noise_level = config.background_noise_level
        temp_builder.preview()
    
    if config and args.save:
        save_config_to_file(config, args.save)
    
    return config

# Example usage functions
def quick_elephant_scene(distance: float = 100, azimuth: float = 0, 
                        noise_level: float = 45) -> SimulationConfig:
    """Quickly create a simple elephant detection scene"""
    return (SceneConfigBuilder()
            .add_elephant(distance=distance, azimuth_deg=azimuth)
            .set_forest_ambient()
            .set_noise_level(noise_level)
            .build())

def create_validation_grid(distances: List[float], azimuths: List[float]) -> List[SimulationConfig]:
    """Create a grid of validation scenarios for systematic testing"""
    
    configs = []
    
    for distance in distances:
        for azimuth in azimuths:
            config = (SceneConfigBuilder()
                     .add_elephant(distance=distance, azimuth_deg=azimuth, 
                                  start_time=2.0)
                     .set_forest_ambient(level_db=20.0)
                     .set_duration(8.0)
                     .set_noise_level(35.0)
                     .build())
            configs.append(config)
    
    return configs

def create_snr_test_series(base_distance: float = 100, 
                          noise_levels: List[float] = None) -> List[SimulationConfig]:
    """Create series of configs with varying SNR for detection threshold testing"""
    
    if noise_levels is None:
        noise_levels = [30, 35, 40, 45, 50, 55, 60]  # dB
    
    configs = []
    
    for noise_level in noise_levels:
        config = (SceneConfigBuilder()
                 .add_elephant(distance=base_distance, azimuth_deg=45)
                 .set_forest_ambient(level_db=25.0)
                 .set_noise_level(noise_level)
                 .set_duration(10.0)
                 .build())
        configs.append(config)
    
    return configs

if __name__ == "__main__":
    # If run as script, use command-line interface
    config = main()
    
    if config:
        print(f"\nFinal configuration ready with {len(config.sources)} sources")
        print("To generate the scene:")
        print("  from acoustic_scene_generator import AcousticSceneGenerator")
        print("  generator = AcousticSceneGenerator()")
        print("  mic_signals, metadata = generator.generate_scene(config)")
    
    # Example of programmatic usage
    print("\nProgrammatic usage examples:")
    print("1. Quick scene: quick_elephant_scene(distance=150, azimuth=30)")
    print("2. Validation grid: create_validation_grid([50,100,200], [0,90,180,270])")
    print("3. SNR testing: create_snr_test_series(base_distance=100)")
    
    # Show preset scenarios
    print("\nPreset scenarios available:")
    scenarios = create_preset_scenarios()
    for name in scenarios.keys():
        print(f"  - {name}")
: elephant <distance> <azimuth> [elevation] [start_time] [volume]")
        
        elif command.startswith('bird'):
            try:
                parts = command.split()[1:]
                distance = float(parts[0])
                azimuth = float(parts[1])
                elevation = float(parts[2]) if len(parts) > 2 else 5.0
                start_time = float(parts[3]) if len(parts) > 3 else 1.0
                volume = float(parts[4]) if len(parts) > 4 else 0.7
                
                builder.add_bird(distance, azimuth, elevation, start_time, volume=volume)
                print(f"Added bird at {distance}m, {azimuth}°")
            except (IndexError, ValueError):
                print("Usage: bird <distance> <azimuth> [elevation] [start_time] [volume]")
        
        elif command.startswith('rain'):
            try:
                parts = command.split()[1:]
                intensity = parts[0] if parts else "moderate"
                level_db = float(parts[1]) if len(parts) > 1 else 35.0
                
                builder.set_rain(intensity, level_db)
                print(f"Added rain: {intensity} ({level_db} dB)")
            except (IndexError, ValueError):
                print("Usage: rain <intensity> [level_db]")
        
        elif command.startswith('wind'):
            try:
                parts = command.split()[1:]
                speed = float(parts[0]) if parts else 15.0
                level_db = float(parts[1]) if len(parts) > 1 else 30.0
                
                builder.set_wind(speed, level_db)
                print(f"Added wind: {speed} km/h ({level_db} dB)")
            except (IndexError, ValueError):
                print("Usage: wind <speed_kmh> [level_db]")
        
        elif command.startswith('duration'):
            try:
                duration = float(command.split()[1])
                builder.set_duration(duration)
                print(f"Set duration to {duration} seconds")
            except (IndexError, ValueError):
                print("Usage
