"""
Hydra Configuration System for Wildlife Acoustic Simulation
Provides structured, reproducible configuration management
"""

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.config_store import ConfigStore
from models import *
from pathlib import Path
import os
from dataclasses import dataclass
from typing import List, Dict, Any

# Create dataclass wrappers for Hydra compatibility
@dataclass
class HydraEnvironmentConfig:
    environment_type: str = "forest_clearing"
    room_size: List[float] = None
    absorption_coefficient: float = 0.02
    max_order: int = 2
    temperature: float = 20.0
    humidity: float = 50.0
    air_absorption: bool = True
    
    def __post_init__(self):
        if self.room_size is None:
            self.room_size = [1000.0, 1000.0, 50.0]

@dataclass 
class HydraProcessingConfig:
    sample_rate: int = 16000
    frame_size: int = 512
    hop_length: int = 256
    window_type: str = "hann"

# Register dataclass versions with Hydra's ConfigStore
cs = ConfigStore.instance()

# Register environment configurations
cs.store(group="environment", name="forest_clearing", node=HydraEnvironmentConfig(
    environment_type="forest_clearing",
    room_size=[1000.0, 1000.0, 50.0],
    absorption_coefficient=0.02,
    max_order=2,
    temperature=25.0,
    humidity=60.0
))

cs.store(group="environment", name="dense_forest", node=HydraEnvironmentConfig(
    environment_type="dense_forest", 
    room_size=[800.0, 800.0, 30.0],
    absorption_coefficient=0.08,
    max_order=3,
    temperature=22.0,
    humidity=75.0
))

cs.store(group="environment", name="savanna", node=HydraEnvironmentConfig(
    environment_type="savanna",
    room_size=[2000.0, 2000.0, 100.0],
    absorption_coefficient=0.01,
    max_order=1,
    temperature=30.0,
    humidity=40.0
))

# Register processing configurations  
cs.store(group="processing", name="standard", node=HydraProcessingConfig(
    sample_rate=16000,
    frame_size=512,
    hop_length=256,
    window_type="hann"
))

cs.store(group="processing", name="high_resolution", node=HydraProcessingConfig(
    sample_rate=48000,
    frame_size=2048,
    hop_length=512,
    window_type="hann"
))

cs.store(group="processing", name="real_time", node=HydraProcessingConfig(
    sample_rate=16000,
    frame_size=256,
    hop_length=128,
    window_type="hann"
))

def convert_hydra_to_pydantic(hydra_config: DictConfig) -> SimulationConfig:
    """Convert Hydra OmegaConf to Pydantic SimulationConfig"""
    
    # Convert environment config
    env_dict = dict(hydra_config.environment)
    if 'environment_type' in env_dict:
        env_dict['environment_type'] = EnvironmentType(env_dict['environment_type'])
    if 'room_size' in env_dict and isinstance(env_dict['room_size'], list):
        env_dict['room_size'] = tuple(env_dict['room_size'])
    
    environment = EnvironmentConfig(**env_dict)
    
    # Convert processing config
    processing = ProcessingConfig(**dict(hydra_config.processing))
    
    # Convert sources
    sources = []
    if 'sources' in hydra_config:
        for source_dict in hydra_config.sources:
            source_data = dict(source_dict)
            if 'position' in source_data and isinstance(source_data['position'], list):
                source_data['position'] = tuple(source_data['position'])
            if 'source_type' in source_data:
                source_data['source_type'] = SourceType(source_data['source_type'])
            sources.append(SoundSource(**source_data))
    
    # Convert ambient conditions - Pydantic expects Dict[str, AmbientCondition] but default YAML has []
    ambient = {}
    if 'ambient' in hydra_config and hydra_config.ambient:
        if isinstance(hydra_config.ambient, dict):
            for condition_name, condition_data in hydra_config.ambient.items():
                ambient_dict = dict(condition_data)
                ambient_dict['name'] = condition_name
                ambient[condition_name] = AmbientCondition(**ambient_dict)
    
    # Build main config
    config_dict = {
        'scene_duration': hydra_config.scene_duration,
        'scene_name': hydra_config.scene_name,
        'environment': environment,
        'processing': processing,
        'sources': sources,
        'ambient': ambient
    }
    
    # Add optional fields if present
    for field in ['description', 'tags', 'background_noise_level', 'datasets']:
        if field in hydra_config:
            config_dict[field] = hydra_config[field]
    
    return SimulationConfig(**config_dict)

class HydraConfigManager:
    """Manages Hydra configurations and provides utilities"""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.ensure_config_structure()
    
    def ensure_config_structure(self):
        """Create configuration directory structure"""
        
        # Create main config directory
        self.config_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        subdirs = [
            "environment",
            "scenarios", 
            "datasets",
            "processing",
            "experiments"
        ]
        
        for subdir in subdirs:
            (self.config_dir / subdir).mkdir(exist_ok=True)
    
    def create_yaml_configs(self):
        """Create YAML configuration files"""
        
        # Main configuration template
        main_config = """
# @package _global_
defaults:
  - environment: forest_clearing
  - processing: standard
  - _self_

# Basic scene parameters
scene_duration: 10.0
background_noise_level: 45.0
scene_name: "default_scene"
description: "Default acoustic scene configuration"
tags: []

# Sources will be defined in scenario configs
sources: []

# Ambient conditions
ambient: {}

# Dataset configurations
datasets: []
"""
        
        with open(self.config_dir / "config.yaml", 'w') as f:
            f.write(main_config.strip())
        
        # Environment configurations
        self._create_environment_configs()
        
        # Scenario configurations
        self._create_scenario_configs()
        
        # Dataset configurations
        self._create_dataset_configs()
        
        # Processing configurations
        self._create_processing_configs()
        
        # Experiment configurations
        self._create_experiment_configs()
    
    def _create_environment_configs(self):
        """Create environment-specific configurations"""
        
        env_configs = {
            "forest_clearing.yaml": """
# @package environment
environment_type: forest_clearing
room_size: [1000.0, 1000.0, 50.0]
absorption_coefficient: 0.02
max_order: 2
temperature: 25.0
humidity: 60.0
air_absorption: true
""",
            "dense_forest.yaml": """
# @package environment  
environment_type: dense_forest
room_size: [800.0, 800.0, 30.0]
absorption_coefficient: 0.08
max_order: 3
temperature: 22.0
humidity: 75.0
air_absorption: true
""",
            "savanna.yaml": """
# @package environment
environment_type: savanna
room_size: [2000.0, 2000.0, 100.0]
absorption_coefficient: 0.01
max_order: 1
temperature: 30.0
humidity: 40.0
air_absorption: true
""",
            "custom.yaml": """
# @package environment
environment_type: custom
room_size: [1500.0, 1500.0, 75.0]
absorption_coefficient: 0.05
max_order: 2
temperature: 20.0
humidity: 50.0
air_absorption: true
"""
        }
        
        env_dir = self.config_dir / "environment"
        for filename, content in env_configs.items():
            with open(env_dir / filename, 'w') as f:
                f.write(content.strip())
    
    def _create_scenario_configs(self):
        """Create predefined scenario configurations"""
        
        scenario_configs = {
            "single_elephant.yaml": """
# @package _global_
defaults:
  - base_config
  - environment: forest_clearing

scene_name: "single_elephant_validation"
description: "Single elephant call for algorithm validation"
scene_duration: 8.0
background_noise_level: 35.0
tags: ["elephant", "validation", "clean"]

sources:
  - audio_file: "sounds/elephant_adult_male.wav"
    position: [160.0, -20.0, 0.0]
    start_time: 7.0
    volume_scale: 1.1
    source_type: "elephant"

ambient:
  forest_base:
    condition_type: "forest_base"
    intensity: "medium"
    level_db: 25.0
  wind:
    condition_type: "wind"
    intensity: 8.0
    level_db: 22.0
""",
            "complex_forest.yaml": """
# @package _global_
defaults:
  - base_config
  - environment: dense_forest

scene_name: "complex_forest_scene"
description: "Complex multi-species forest environment"
scene_duration: 18.0
background_noise_level: 48.0
tags: ["multi-species", "complex", "forest"]

sources:
  - audio_file: "sounds/elephant_call.wav"
    position: [120.0, 45.0, 0.0]
    start_time: 4.0
    volume_scale: 0.9
    source_type: "elephant"
  
  - audio_file: "sounds/hornbill_call.wav"
    position: [60.0, -30.0, 12.0]
    start_time: 1.0
    volume_scale: 0.7
    source_type: "bird"
  
  - audio_file: "sounds/langur_alarm.wav"
    position: [80.0, 100.0, 15.0]
    start_time: 8.0
    volume_scale: 0.8
    source_type: "monkey"
  
  - audio_file: "sounds/vehicle_distant.wav"
    position: [0.0, -400.0, 0.0]
    start_time: 0.0
    volume_scale: 0.4
    source_type: "vehicle"

ambient:
  rain:
    condition_type: "rain"
    intensity: "light"
    level_db: 32.0
  wind:
    condition_type: "wind"
    intensity: 15.0
    level_db: 28.0
  forest_base:
    condition_type: "forest_base"
    intensity: "medium"
    level_db: 30.0
""",
            "noisy_challenge.yaml": """
# @package _global_
defaults:
  - base_config
  - environment: forest_clearing

scene_name: "noisy_challenge_scenario"
description: "High-noise environment for detection limits testing"
scene_duration: 20.0
background_noise_level: 55.0
tags: ["noisy", "challenge", "detection-limits"]

sources:
  - audio_file: "sounds/elephant_distant.wav"
    position: [350.0, 100.0, 0.0]
    start_time: 5.0
    volume_scale: 0.8
    source_type: "elephant"
  
  - audio_file: "sounds/truck_passing.wav"
    position: [120.0, -150.0, 0.0]
    start_time: 0.0
    volume_scale: 1.5
    source_type: "vehicle"
  
  - audio_file: "sounds/chainsaw.wav"
    position: [200.0, 200.0, 0.0]
    start_time: 8.0
    volume_scale: 1.2
    source_type: "machinery"
  
  - audio_file: "sounds/bird_chorus.wav"
    position: [80.0, 120.0, 10.0]
    start_time: 2.0
    volume_scale: 0.9
    source_type: "bird"

ambient:
  rain:
    condition_type: "rain"
    intensity: "moderate"
    level_db: 40.0
  wind:
    condition_type: "wind"
    intensity: 25.0
    level_db: 35.0
  forest_base:
    condition_type: "forest_base"
    intensity: "high"
    level_db: 35.0
"""
        }
        
        scenario_dir = self.config_dir / "scenarios"
        for filename, content in scenario_configs.items():
            with open(scenario_dir / filename, 'w') as f:
                f.write(content.strip())
    
    def _create_dataset_configs(self):
        """Create dataset configuration files"""
        
        dataset_configs = {
            "kaggle_datasets.yaml": """
# Audio datasets from Kaggle
datasets:
  - dataset_name: "ESC-50"
    source_type: "kaggle"
    path_or_id: "karoldvl/esc50-dataset" 
    audio_extensions: [".wav"]
    metadata_file: "meta/esc50.csv"
    cache_dir: "./audio_cache/esc50"
  
  - dataset_name: "BirdCLEF-2023"
    source_type: "kaggle"
    path_or_id: "competitions/birdclef-2023"
    audio_extensions: [".ogg", ".mp3"]
    metadata_file: "train_metadata.csv"
    cache_dir: "./audio_cache/birdclef2023"
  
  - dataset_name: "Freesound Audio Tagging"
    source_type: "kaggle" 
    path_or_id: "c/freesound-audio-tagging"
    audio_extensions: [".wav"]
    metadata_file: "train.csv"
    cache_dir: "./audio_cache/freesound"

  - dataset_name: "Animal Sound Classification"
    source_type: "kaggle"
    path_or_id: "mmoreaux/audio-cats-and-dogs"
    audio_extensions: [".wav", ".mp3"]
    cache_dir: "./audio_cache/animal_sounds"
""",
            "local_datasets.yaml": """
# Local audio datasets
datasets:
  - dataset_name: "Elephant Recordings"
    source_type: "local"
    path_or_id: "./sounds/elephants/"
    audio_extensions: [".wav", ".mp3", ".flac"]
    metadata_file: "./sounds/elephants/metadata.csv"
    cache_dir: "./sounds/elephants/"
  
  - dataset_name: "Forest Birds"
    source_type: "local"
    path_or_id: "./sounds/birds/"
    audio_extensions: [".wav", ".mp3"]
    metadata_file: "./sounds/birds/species_info.json"
    cache_dir: "./sounds/birds/"
  
  - dataset_name: "Environmental Sounds"
    source_type: "local"
    path_or_id: "./sounds/environment/"
    audio_extensions: [".wav", ".flac"]
    cache_dir: "./sounds/environment/"
""",
            "google_drive.yaml": """
# Google Drive datasets
datasets:
  - dataset_name: "Wildlife Audio Collection"
    source_type: "google_drive"
    path_or_id: "1A2B3C4D5E6F7G8H9I0J"  # Google Drive folder ID
    audio_extensions: [".wav", ".mp3", ".m4a"]
    metadata_file: "collection_metadata.csv"
    cache_dir: "./audio_cache/wildlife_drive"
  
  - dataset_name: "Research Team Recordings"
    source_type: "google_drive"
    path_or_id: "your_shared_folder_id_here"
    audio_extensions: [".wav", ".flac"]
    cache_dir: "./audio_cache/team_recordings"
"""
        }
        
        dataset_dir = self.config_dir / "datasets"
        for filename, content in dataset_configs.items():
            with open(dataset_dir / filename, 'w') as f:
                f.write(content.strip())
    
    def _create_processing_configs(self):
        """Create processing configuration files"""
        
        processing_configs = {
            "standard.yaml": """
# @package processing
# Standard processing for most applications
sample_rate: 16000
frame_size: 512
hop_length: 256
window_type: "hann"
""",
            "high_resolution.yaml": """
# @package processing
# High resolution for detailed analysis
sample_rate: 48000
frame_size: 2048
hop_length: 512
window_type: "hann"
""",
            "real_time.yaml": """
# @package processing
# Optimized for real-time processing
sample_rate: 16000
frame_size: 256
hop_length: 128
window_type: "hann"
""",
            "low_latency.yaml": """
# @package processing
# Minimal latency configuration
sample_rate: 16000
frame_size: 128
hop_length: 64
window_type: "hann"
"""
        }
        
        processing_dir = self.config_dir / "processing"
        for filename, content in processing_configs.items():
            with open(processing_dir / filename, 'w') as f:
                f.write(content.strip())
    
    def _create_experiment_configs(self):
        """Create experiment configuration files"""
        
        experiment_configs = {
            "doa_validation.yaml": """
# @package _global_
defaults:
  - scenarios/single_elephant
  - processing: high_resolution

# Override for DOA validation experiment
scene_name: "doa_validation_experiment"
description: "Systematic DOA algorithm validation"
tags: ["experiment", "doa", "validation"]

# Multiple test positions
hydra:
  mode: MULTIRUN
  sweeper:
    _target_: hydra._internal.BasicSweeper
    max_batch_size: null
    params:
      sources.0.position: "[50,0,0],[100,0,0],[200,0,0],[100,100,0]"
      background_noise_level: "30,35,40,45,50"
""",
            "range_accuracy.yaml": """
# @package _global_
defaults:
  - scenarios/single_elephant
  - environment: savanna

scene_name: "range_accuracy_test"
description: "DOA accuracy vs distance testing"
tags: ["experiment", "range", "accuracy"]

hydra:
  mode: MULTIRUN
  sweeper:
    _target_: hydra._internal.BasicSweeper
    max_batch_size: null
    params:
      sources.0.position: "[[50,0,0],[100,0,0],[200,0,0],[400,0,0],[800,0,0]]"
      sources.0.start_time: "1.0,2.0,3.0"
""",
            "multi_source_stress.yaml": """
# @package _global_
defaults:
  - scenarios/complex_forest
  - processing: standard

scene_name: "multi_source_stress_test"
description: "Stress testing with many simultaneous sources"
tags: ["experiment", "stress-test", "multi-source"]

scene_duration: 30.0
background_noise_level: 50.0

# Add more sources for stress testing
sources:
  - audio_file: "sounds/elephant_1.wav"
    position: [100.0, 0.0, 0.0]
    start_time: 2.0
    volume_scale: 1.0
    source_type: "elephant"
  - audio_file: "sounds/elephant_2.wav"
    position: [150.0, 100.0, 0.0]
    start_time: 5.0
    volume_scale: 0.9
    source_type: "elephant"
  - audio_file: "sounds/bird_1.wav"
    position: [50.0, 50.0, 8.0]
    start_time: 1.0
    volume_scale: 0.7
    source_type: "bird"
  - audio_file: "sounds/bird_2.wav"
    position: [80.0, -40.0, 12.0]
    start_time: 8.0
    volume_scale: 0.6
    source_type: "bird"
  - audio_file: "sounds/monkey_1.wav"
    position: [120.0, 80.0, 15.0]
    start_time: 12.0
    volume_scale: 0.8
    source_type: "monkey"
"""
        }
        
        experiment_dir = self.config_dir / "experiments"
        for filename, content in experiment_configs.items():
            with open(experiment_dir / filename, 'w') as f:
                f.write(content.strip())

@hydra.main(version_base=None, config_path="configs", config_name="config")
def run_simulation(cfg: DictConfig) -> None:
    """Main simulation runner with Hydra configuration"""
    
    # Convert OmegaConf to Pydantic model for validation
    try:
        config = convert_hydra_to_pydantic(cfg)
        print(f"✓ Configuration validated successfully")
        print(f"  Scene: {config.scene_name}")
        print(f"  Duration: {config.scene_duration}s")
        print(f"  Sources: {config.total_sources}")
        print(f"  Environment: {config.environment.environment_type}")
        
        # Import and run simulation
        from acoustic_scene_generator import WildlifeAcousticSimulator
        
        simulator = WildlifeAcousticSimulator()
        mic_signals, metadata = simulator.simulate_scene(config)
        
        print(f"✓ Simulation completed")
        print(f"  Output shape: {mic_signals.shape}")
        print(f"  Peak amplitude: {metadata['mic_signals_stats']['peak_amplitude']:.3f}")
        
        # Save results
        output_dir = f"outputs/{config.scene_name}"
        from acoustic_scene_generator import save_simulation_results
        save_simulation_results(mic_signals, metadata, output_dir, config.scene_name)
        
    except Exception as e:
        print(f"✗ Simulation failed: {e}")
        raise

def create_config_templates():
    """Utility function to create all configuration templates"""
    
    manager = HydraConfigManager()
    manager.create_yaml_configs()
    
    print("Configuration templates created successfully!")
    print("Directory structure:")
    print("configs/")
    print("├── config.yaml                 # Main configuration")
    print("├── environment/                # Environment presets")
    print("│   ├── forest_clearing.yaml")
    print("│   ├── dense_forest.yaml") 
    print("│   ├── savanna.yaml")
    print("│   └── custom.yaml")
    print("├── scenarios/                  # Predefined scenarios")
    print("│   ├── single_elephant.yaml")
    print("│   ├── elephant_herd.yaml")
    print("│   ├── complex_forest.yaml")
    print("│   └── noisy_challenge.yaml")
    print("├── datasets/                   # Dataset configurations")
    print("│   ├── kaggle_datasets.yaml")
    print("│   ├── local_datasets.yaml")
    print("│   └── google_drive.yaml")
    print("├── processing/                 # Processing parameters")
    print("│   ├── standard.yaml")
    print("│   ├── high_resolution.yaml")
    print("│   ├── real_time.yaml")
    print("│   └── low_latency.yaml")
    print("└── experiments/                # Experiment configurations")
    print("    ├── doa_validation.yaml")
    print("    ├── range_accuracy.yaml")
    print("    └── multi_source_stress.yaml")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "create-configs":
        create_config_templates()
    else:
        # Run normal Hydra application
        run_simulation()