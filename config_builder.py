"""
Enhanced Configuration Builder for Pyroomacoustics-based Acoustic Scene Simulation
"""

from acoustic_scene_generator import *
import json
from typing import Dict, List
import argparse

class EnhancedSceneConfigBuilder:
    """Enhanced configuration builder for pyroomacoustics-based simulation"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset to default configuration"""
        self.duration = 10.0
        self.sample_rate = 16000
        self.sources = []
        self.ambient = {}
        self.noise_level = 45.0
        self.environment = EnvironmentConfig()
        self.frame_size = 512
        self.hop_length = 256
    
    def set_environment(self, room_size: Tuple[float, float, float] = (1000, 1000, 50),
                       absorption: float = 0.02, max_order: int = 2, 
                       temperature: float = 20.0, humidity: float = 50.0):
        """Configure the acoustic environment"""
        self.environment = EnvironmentConfig(
            room_size=room_size,
            absorption_coefficient=absorption,
            max_order=max_order,
            temperature=temperature,
            humidity=humidity,
            air_absorption=True
        )
        return self
    
    def add_elephant(self, distance: float, azimuth_deg: float, 
                    elevation: float = 0.0, start_time: float = 2.0,
                    audio_file: str = "elephant_call.wav", volume: float = 1.0,
                    duration_override: float = None):
        """Add elephant source with enhanced parameters"""
        
        azimuth_rad = np.radians(azimuth_deg)
        x = distance * np.cos(azimuth_rad)
        y = distance * np.sin(azimuth_rad)
        z = elevation
        
        source = SoundSource(
            audio_file=audio_file,
            position=(x, y, z),
            start_time=start_time,
            volume_scale=volume,
            source_type="elephant",
            duration_override=duration_override
        )
        
        self.sources.append(source)
        return self
    
    def add_bird(self, distance: float, azimuth_deg: float,
                elevation: float = 8.0, start_time: float = 1.0,
                audio_file: str = "bird_call.wav", volume: float = 0.7,
                duration_override: float = None):
        """Add bird source with realistic elevation"""
        
        azimuth_rad = np.radians(azimuth_deg)
        x = distance * np.cos(azimuth_rad)
        y = distance * np.sin(azimuth_rad)
        z = elevation
        
        source = SoundSource(
            audio_file=audio_file,
            position=(x, y, z),
            start_time=start_time,
            volume_scale=volume,
            source_type="bird",
            duration_override=duration_override
        )
        
        self.sources.append(source)
        return self
    
    def add_vehicle(self, distance: float, azimuth_deg: float,
                   start_time: float = 0.0, audio_file: str = "vehicle.wav",
                   volume: float = 0.8, duration_override: float = None):
        """Add vehicle source (ground level)"""
        
        azimuth_rad = np.radians(azimuth_deg)
        x = distance * np.cos(azimuth_rad)
        y = distance * np.sin(azimuth_rad)
        z = 0.0
        
        source = SoundSource(
            audio_file=audio_file,
            position=(x, y, z),
            start_time=start_time,
            volume_scale=volume,
            source_type="vehicle",
            duration_override=duration_override
        )
        
        self.sources.append(source)
        return self
    
    def add_custom_source(self, source_type: str, distance: float, 
                         azimuth_deg: float, elevation: float = 0.0,
                         start_time: float = 0.0, audio_file: str = "",
                         volume: float = 1.0, duration_override: float = None):
        """Add custom source type with full parameter control"""
        
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
            source_type=source_type,
            duration_override=duration_override
        )
        
        self.sources.append(source)
        return self
    
    def set_forest_clearing(self, size_km: float = 1.0, height_m: float = 50.0):
        """Configure as forest clearing environment"""
        size_m = size_km * 1000
        self.environment = EnvironmentConfig(
            room_size=(size_m, size_m, height_m),
            absorption_coefficient=0.02,  # Very low for outdoor
            max_order=2,  # Limited reflections
            temperature=25.0,
            humidity=60.0,
            air_absorption=True
        )
        return self
    
    def set_dense_forest(self, size_km: float = 0.5, height_m: float = 30.0):
        """Configure as dense forest with more absorption"""
        size_m = size_km * 1000
        self.environment = EnvironmentConfig(
            room_size=(size_m, size_m, height_m),
            absorption_coefficient=0.08,  # Higher absorption for dense vegetation
            max_order=3,  # More reflections from trees
            temperature=22.0,
            humidity=75.0,  # Higher humidity in forest
            air_absorption=True
        )
        return self
    
    def set_savanna(self, size_km: float = 2.0, height_m: float = 100.0):
        """Configure as open savanna environment"""
        size_m = size_km * 1000
        self.environment = EnvironmentConfig(
            room_size=(size_m, size_m, height_m),
            absorption_coefficient=0.01,  # Very open space
            max_order=1,  # Minimal reflections
            temperature=30.0,
            humidity=40.0,  # Drier climate
            air_absorption=True
        )
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
        """Add forest base ambient sounds"""
        self.ambient['forest_base'] = AmbientCondition('forest_base', 'medium', level_db)
        return self
    
    def set_processing_params(self, frame_size: int = 512, hop_length: int = 256):
        """Set STFT processing parameters"""
        self.frame_size = frame_size
        self.hop_length = hop_length
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
            environment=self.environment,
            ambient=self.ambient.copy(),
            background_noise_level=self.noise_level,
            frame_size=self.frame_size,
            hop_length=self.hop_length
        )
    
    def preview(self):
        """Enhanced preview with environment information"""
        print("=== Enhanced Scene Configuration Preview ===")
        print(f"Duration: {self.duration} seconds")
        print(f"Sample rate: {self.sample_rate} Hz")
        print(f"Frame size: {self.frame_size}, Hop: {self.hop_length}")
        print(f"Background noise: {self.noise_level} dB")
        
        print(f"\nEnvironment:")
        print(f"  Room size: {self.environment.room_size} m")
        print(f"  Absorption: {self.environment.absorption_coefficient}")
        print(f"  Max order: {self.environment.max_order}")
        print(f"  Temperature: {self.environment.temperature}°C")
        print(f"  Humidity: {self.environment.humidity}%")
        
        print(f"\nSources ({len(self.sources)}):")
        for i, source in enumerate(self.sources):
            x, y, z = source.position
            distance = np.sqrt(x**2 + y**2)
            azimuth = np.degrees(np.arctan2(y, x))
            if azimuth < 0:
                azimuth += 360
                
            duration_info = f", dur={source.duration_override:.1f}s" if source.duration_override else ""
            print(f"  {i+1}. {source.source_type}: {distance:.0f}m @ {azimuth:.0f}° "
                  f"(elev={z:.1f}m, t={source.start_time:.1f}s, vol={source.volume_scale:.1f}{duration_info})")
        
        print(f"\nAmbient conditions: {list(self.ambient.keys())}")
        for condition_type, condition in self.ambient.items():
            print(f"  {condition_type}: {condition.intensity} ({condition.level_db:.0f} dB)")

def create_enhanced_preset_scenarios():
    """Create enhanced preset scenarios using new features"""
    
    scenarios = {}
    
    # Simple elephant in forest clearing
    scenarios['forest_elephant'] = (
        EnhancedSceneConfigBuilder()
        .set_forest_clearing(size_km=1.0)
        .add_elephant(distance=120, azimuth_deg=45, start_time=2.0)
        .set_forest_ambient(level_db=22.0)
        .set_wind(speed_kmh=8, level_db=25.0)
        .set_duration(10.0)
        .set_noise_level(38.0)
    )
    
    # Savanna elephant with distant sounds
    scenarios['savanna_elephant'] = (
        EnhancedSceneConfigBuilder()
        .set_savanna(size_km=2.0)
        .add_elephant(distance=200, azimuth_deg=0, start_time=3.0, volume=1.2)
        .add_bird(distance=80, azimuth_deg=135, elevation=5.0, start_time=1.0)
        .add_vehicle(distance=500, azimuth_deg=270, start_time=0.0, volume=0.4)
        .set_wind(speed_kmh=15, level_db=28.0)
        .set_duration(12.0)
        .set_noise_level(40.0)
    )
    
    # Dense forest with multiple animals
    scenarios['dense_forest_multi'] = (
        EnhancedSceneConfigBuilder()
        .set_dense_forest(size_km=0.8)
        .add_elephant(distance=100, azimuth_deg=30, start_time=4.0)
        .add_custom_source("monkey", distance=45, azimuth_deg=150, elevation=15.0, 
                          start_time=1.0, volume=0.8)
        .add_bird(distance=25, azimuth_deg=60, elevation=12.0, start_time=0.5)
        .add_bird(distance=35, azimuth_deg=300, elevation=8.0, start_time=6.0)
        .set_rain(intensity="light", level_db=30.0)
        .set_forest_ambient(level_db=28.0)
        .set_duration(15.0)
        .set_noise_level(45.0)
    )
    
    # Elephant herd scenario
    scenarios['elephant_herd_realistic'] = (
        EnhancedSceneConfigBuilder()
        .set_forest_clearing(size_km=1.5)
        .add_elephant(distance=150, azimuth_deg=0, start_time=2.0, 
                     audio_file="elephant_adult_female.wav", volume=1.0)
        .add_elephant(distance=140, azimuth_deg=20, start_time=4.5,
                     audio_file="elephant_calf.wav", volume=0.6, duration_override=2.5)
        .add_elephant(distance=160, azimuth_deg=-15, start_time=7.0,
                     audio_file="elephant_adult_male.wav", volume=1.1)
        .set_forest_ambient(level_db=20.0)
        .set_wind(speed_kmh=5, level_db=22.0)
        .set_duration(12.0)
        .set_noise_level(35.0)
    )
    
    # High-noise challenge scenario
    scenarios['noisy_challenge_realistic'] = (
        EnhancedSceneConfigBuilder()
        .set_forest_clearing(size_km=0.8)
        .add_elephant(distance=300, azimuth_deg=60, start_time=3.0, volume=0.9)
        .add_vehicle(distance=120, azimuth_deg=180, start_time=0.0, volume=1.5, 
                    duration_override=8.0)
        .add_custom_source("chainsaw", distance=200, azimuth_deg=270, 
                          start_time=5.0, volume=1.2, duration_override=4.0)
        .add_bird(distance=30, azimuth_deg=90, start_time=1.0, volume=0.9)
        .set_rain(intensity="moderate", level_db=40.0)
        .set_wind(speed_kmh=22, level_db=35.0)
        .set_forest_ambient(level_db=32.0)
        .set_duration(15.0)
        .set_noise_level(55.0)
    )
    
    # Algorithm validation scenario
    scenarios['validation_precise'] = (
        EnhancedSceneConfigBuilder()
        .set_forest_clearing(size_km=0.5)
        .add_elephant(distance=100, azimuth_deg=90, start_time=2.0)  # Due east
        .set_forest_ambient(level_db=18.0)
        .set_duration(8.0)
        .set_noise_level(30.0)
        .set_processing_params(frame_size=1024, hop_length=512)  # High resolution
    )
    
    # Multi-range validation
    scenarios['range_validation'] = (
        EnhancedSceneConfigBuilder()
        .set_savanna(size_km=1.5)
        .add_elephant(distance=50, azimuth_deg=0, start_time=1.0,
                     audio_file="elephant_close.wav", volume=0.8)
        .add_elephant(distance=150, azimuth_deg=120, start_time=4.0,
                     audio_file="elephant_medium.wav", volume=1.0)
        .add_elephant(distance=400, azimuth_deg=240, start_time=7.0,
                     audio_file="elephant_distant.wav", volume=1.3)
        .set_wind(speed_kmh=10, level_db=25.0)
        .set_duration(12.0)
        .set_noise_level(42.0)
    )
    
    return scenarios

def batch_generate_validation_dataset(output_dir: str, scenarios: Dict, 
                                     variations_per_scenario: int = 10):
    """Generate a comprehensive validation dataset with multiple variations"""
    
    from acoustic_scene_generator import WildlifeAcousticSimulator
    import random
    from pathlib import Path
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    simulator = WildlifeAcousticSimulator()
    dataset_info = []
    scene_counter = 0
    
    print(f"Generating validation dataset with {len(scenarios)} scenario types")
    print(f"Creating {variations_per_scenario} variations per scenario")
    
    for scenario_name, builder in scenarios.items():
        print(f"\nGenerating {scenario_name} scenarios...")
        
        for variation in range(variations_per_scenario):
            # Create base configuration
            config = builder.build()
            
            # Add random variations
            config = add_realistic_variations(config, variation)
            
            try:
                # Generate scene
                mic_signals, metadata = simulator.simulate_scene(config)
                
                # Validate with DOA algorithms
                validation = simulator.validate_with_doa_algorithms(mic_signals, config)
                metadata['doa_validation'] = validation
                
                # Save results
                scene_name = f"scene_{scene_counter:04d}_{scenario_name}_v{variation:02d}"
                save_simulation_results(mic_signals, metadata, output_dir, scene_name)
                
                # Track dataset info
                scene_info = {
                    'scene_id': scene_name,
                    'scenario_type': scenario_name,
                    'variation': variation,
                    'num_sources': len(config.sources),
                    'elephant_count': sum(1 for s in config.sources if s.source_type == 'elephant'),
                    'duration': config.scene_duration,
                    'noise_level': config.background_noise_level,
                    'environment': config.environment.__dict__,
                    'validation_success': {k: 'error' not in v for k, v in validation.items()}
                }
                dataset_info.append(scene_info)
                
                print(f"  Generated {scene_name}")
                scene_counter += 1
                
            except Exception as e:
                print(f"  Failed to generate {scenario_name} variation {variation}: {e}")
                continue
    
    # Save dataset summary
    with open(output_path / "dataset_summary.json", 'w') as f:
        json.dump(dataset_info, f, indent=2, default=str)
    
    # Generate analysis report
    generate_dataset_report(dataset_info, output_path)
    
    print(f"\nDataset generation complete!")
    print(f"Generated {len(dataset_info)} scenes in {output_dir}")
    print(f"Dataset summary saved to dataset_summary.json")
    
    return dataset_info

def add_realistic_variations(config: SimulationConfig, seed: int) -> SimulationConfig:
    """Add realistic variations to configuration"""
    random.seed(seed)
    np.random.seed(seed)
    
    # Vary source positions (±10% of distance)
    for source in config.sources:
        original_pos = np.array(source.position)
        distance = np.linalg.norm(original_pos[:2])
        
        # Add position jitter
        jitter_amount = distance * 0.1
        position_jitter = np.random.uniform(-jitter_amount, jitter_amount, 3)
        position_jitter[2] *= 0.3  # Less vertical jitter
        
        new_pos = original_pos + position_jitter
        source.position = tuple(new_pos)
        
        # Vary start time (±1.5 seconds)
        time_jitter = np.random.uniform(-1.5, 1.5)
        source.start_time = max(0, source.start_time + time_jitter)
        
        # Vary volume (±15%)
        volume_jitter = np.random.uniform(0.85, 1.15)
        source.volume_scale *= volume_jitter
    
    # Vary environment slightly
    config.environment.temperature += np.random.uniform(-5, 5)
    config.environment.humidity += np.random.uniform(-10, 10)
    config.environment.humidity = np.clip(config.environment.humidity, 10, 95)
    
    # Vary noise level (±3 dB)
    config.background_noise_level += np.random.uniform(-3, 3)
    
    # Vary ambient levels
    for condition in config.ambient.values():
        condition.level_db += np.random.uniform(-2, 2)
    
    return config

def generate_dataset_report(dataset_info: List[Dict], output_path: Path):
    """Generate analysis report for the dataset"""
    
    report = {
        'total_scenes': len(dataset_info),
        'scenario_breakdown': {},
        'source_statistics': {},
        'environment_statistics': {},
        'validation_success_rates': {}
    }
    
    # Analyze by scenario type
    for scene in dataset_info:
        scenario = scene['scenario_type']
        if scenario not in report['scenario_breakdown']:
            report['scenario_breakdown'][scenario] = 0
        report['scenario_breakdown'][scenario] += 1
    
    # Source statistics
    total_sources = sum(scene['num_sources'] for scene in dataset_info)
    total_elephants = sum(scene['elephant_count'] for scene in dataset_info)
    
    report['source_statistics'] = {
        'total_sources': total_sources,
        'total_elephants': total_elephants,
        'avg_sources_per_scene': total_sources / len(dataset_info),
        'elephant_percentage': (total_elephants / total_sources) * 100
    }
    
    # Validation success rates
    validation_methods = ['MUSIC', 'SRP-PHAT', 'FRIDA']
    for method in validation_methods:
        successes = sum(1 for scene in dataset_info 
                       if scene['validation_success'].get(method, False))
        report['validation_success_rates'][method] = (successes / len(dataset_info)) * 100
    
    # Save report
    with open(output_path / "dataset_report.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nDataset Analysis Report:")
    print(f"Total scenes: {report['total_scenes']}")
    print(f"Average sources per scene: {report['source_statistics']['avg_sources_per_scene']:.1f}")
    print(f"Elephant percentage: {report['source_statistics']['elephant_percentage']:.1f}%")
    print("Validation success rates:")
    for method, rate in report['validation_success_rates'].items():
        print(f"  {method}: {rate:.1f}%")

def main():
    """Enhanced main function with new options"""
    
    parser = argparse.ArgumentParser(description="Enhanced Acoustic Scene Configuration Builder")
    parser.add_argument("--preset", choices=list(create_enhanced_preset_scenarios().keys()),
                       help="Use an enhanced preset scenario")
    parser.add_argument("--generate-dataset", action="store_true",
                       help="Generate comprehensive validation dataset")
    parser.add_argument("--dataset-size", type=int, default=10,
                       help="Variations per scenario for dataset generation")
    parser.add_argument("--output-dir", type=str, default="validation_dataset",
                       help="Output directory for dataset")
    parser.add_argument("--save", type=str, help="Save configuration to file")
    parser.add_argument("--load", type=str, help="Load configuration from file")
    parser.add_argument("--preview", action="store_true", help="Preview configuration")
    parser.add_argument("--list-presets", action="store_true", 
                       help="List available preset scenarios")
    
    args = parser.parse_args()
    
    scenarios = create_enhanced_preset_scenarios()
    
    if args.list_presets:
        print("Available enhanced preset scenarios:")
        for name, builder in scenarios.items():
            print(f"\n{name.upper()}:")
            builder.preview()
        return
    
    if args.generate_dataset:
        print("Generating comprehensive validation dataset...")
        dataset_info = batch_generate_validation_dataset(
            args.output_dir, scenarios, args.dataset_size
        )
        return dataset_info
    
    # Handle single configuration (existing logic)
    config = None
    
    if args.load:
        config = load_config_from_file(args.load)
        print(f"Loaded configuration from {args.load}")
    
    elif args.preset:
        builder = scenarios[args.preset]
        config = builder.build()
        print(f"Using enhanced preset: {args.preset}")
        if args.preview:
            builder.preview()
    
    if config and args.save:
        save_config_to_file(config, args.save)
    
    return config

if __name__ == "__main__":
    main()