"""
Practical examples for generating synthetic acoustic monitoring data
"""

from acoustic_scene_generator import *
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def create_elephant_detection_scenario():
    """Create a scenario focused on elephant detection with various distractors"""
    
    sources = [
        # Primary target: Elephant at various distances and directions
        SoundSource(
            audio_file="sounds/elephant_rumble_1.wav",
            position=(120.0, 50.0, 0.0),  # 130m NNE
            start_time=3.0,
            volume_scale=1.0,
            source_type="elephant"
        ),
        
        # Distractor sounds
        SoundSource(
            audio_file="sounds/bird_hornbill.wav",
            position=(30.0, -25.0, 8.0),  # Close bird
            start_time=1.5,
            volume_scale=0.6,
            source_type="bird"
        ),
        
        SoundSource(
            audio_file="sounds/monkey_langur.wav", 
            position=(-60.0, 80.0, 12.0),  # Monkey in trees
            start_time=5.5,
            volume_scale=0.8,
            source_type="monkey"
        ),
        
        SoundSource(
            audio_file="sounds/vehicle_distant.wav",
            position=(0.0, -300.0, 0.0),  # Distant road
            start_time=0.0,
            volume_scale=0.4,
            source_type="vehicle"
        )
    ]
    
    ambient = {
        'wind': AmbientCondition('wind', 18.0, 32.0),  # Moderate wind
        'forest_base': AmbientCondition('forest_base', 'medium', 28.0)
    }
    
    return SimulationConfig(
        scene_duration=15.0,
        sources=sources,
        ambient=ambient,
        background_noise_level=48.0  # Slightly noisy environment
    )

def create_multi_elephant_scenario():
    """Multiple elephants for testing source separation"""
    
    sources = [
        SoundSource(
            audio_file="sounds/elephant_call_adult.wav",
            position=(200.0, 0.0, 0.0),  # Adult elephant due North
            start_time=2.0,
            volume_scale=1.0,
            source_type="elephant"
        ),
        
        SoundSource(
            audio_file="sounds/elephant_calf.wav",
            position=(180.0, 30.0, 0.0),  # Calf nearby
            start_time=4.0,
            volume_scale=0.7,
            source_type="elephant"
        ),
        
        SoundSource(
            audio_file="sounds/elephant_trumpet.wav",
            position=(150.0, -50.0, 0.0),  # Different elephant, agitated
            start_time=7.0,
            volume_scale=1.2,
            source_type="elephant"
        )
    ]
    
    ambient = {
        'forest_base': AmbientCondition('forest_base', 'low', 25.0)
    }
    
    return SimulationConfig(
        scene_duration=12.0,
        sources=sources,
        ambient=ambient,
        background_noise_level=42.0  # Quiet environment
    )

def create_noisy_environment_scenario():
    """High-noise scenario to test detection limits"""
    
    sources = [
        SoundSource(
            audio_file="sounds/elephant_distant.wav",
            position=(350.0, 100.0, 0.0),  # Very distant elephant
            start_time=3.0,
            volume_scale=0.8,
            source_type="elephant"
        ),
        
        # Multiple noise sources
        SoundSource(
            audio_file="sounds/truck_passing.wav",
            position=(-50.0, -150.0, 0.0),  # Truck on road
            start_time=1.0,
            volume_scale=1.5,
            source_type="vehicle"
        ),
        
        SoundSource(
            audio_file="sounds/chainsaw.wav",
            position=(200.0, -200.0, 0.0),  # Human activity
            start_time=5.0,
            volume_scale=1.1,
            source_type="human_activity"
        ),
        
        SoundSource(
            audio_file="sounds/bird_chorus.wav",
            position=(80.0, 120.0, 15.0),  # Bird dawn chorus
            start_time=0.0,
            volume_scale=0.9,
            source_type="birds"
        )
    ]
    
    ambient = {
        'rain': AmbientCondition('rain', 'moderate', 40.0),
        'wind': AmbientCondition('wind', 25.0, 35.0),  # Strong wind
        'forest_base': AmbientCondition('forest_base', 'medium', 30.0)
    }
    
    return SimulationConfig(
        scene_duration=10.0,
        sources=sources,
        ambient=ambient,
        background_noise_level=55.0  # High background noise
    )

def create_validation_scenario():
    """Simple scenario with known sources for algorithm validation"""
    
    sources = [
        # Single elephant at known location for precise validation
        SoundSource(
            audio_file="sounds/elephant_reference.wav",
            position=(100.0, 0.0, 0.0),  # Exactly 100m North
            start_time=2.0,
            volume_scale=1.0,
            source_type="elephant"
        )
    ]
    
    ambient = {
        'forest_base': AmbientCondition('forest_base', 'minimal', 20.0)
    }
    
    return SimulationConfig(
        scene_duration=8.0,
        sources=sources,
        ambient=ambient,
        background_noise_level=35.0  # Very clean environment
    )

def generate_training_dataset(output_dir: str, num_scenes: int = 100):
    """Generate a comprehensive training dataset"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    generator = AcousticSceneGenerator()
    dataset_info = []
    
    scenario_generators = [
        create_elephant_detection_scenario,
        create_multi_elephant_scenario, 
        create_noisy_environment_scenario,
        create_validation_scenario
    ]
    
    for scene_id in range(num_scenes):
        # Randomly select scenario type
        scenario_func = np.random.choice(scenario_generators)
        config = scenario_func()
        
        # Add some randomization
        config = randomize_config(config, scene_id)
        
        try:
            # Generate scene
            mic_signals, metadata = generator.generate_scene(config)
            
            # Save results
            scene_name = f"scene_{scene_id:04d}"
            save_simulation_results(mic_signals, metadata, output_dir, scene_name)
            
            # Track dataset info
            scene_info = {
                'scene_id': scene_name,
                'scenario_type': scenario_func.__name__,
                'num_sources': len(config.sources),
                'elephant_count': sum(1 for s in config.sources if s.source_type == 'elephant'),
                'duration': config.scene_duration,
                'noise_level': config.background_noise_level
            }
            dataset_info.append(scene_info)
            
            print(f"Generated {scene_name}: {scenario_func.__name__}")
            
        except FileNotFoundError as e:
            print(f"Skipping scene {scene_id}: {e}")
            continue
    
    # Save dataset summary
    with open(output_path / "dataset_info.json", 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    print(f"\nDataset generation complete!")
    print(f"Generated {len(dataset_info)} scenes in {output_dir}")
    return dataset_info

def randomize_config(config: SimulationConfig, seed: int = None) -> SimulationConfig:
    """Add randomization to base configuration"""
    if seed is not None:
        np.random.seed(seed)
    
    # Randomize source positions slightly
    for source in config.sources:
        # Add position jitter (±20% of original distance)
        original_pos = np.array(source.position)
        distance = np.linalg.norm(original_pos[:2])  # 2D distance
        jitter_amount = distance * 0.2
        
        jitter = np.random.uniform(-jitter_amount, jitter_amount, 3)
        jitter[2] *= 0.5  # Less jitter in elevation
        
        new_pos = original_pos + jitter
        source.position = tuple(new_pos)
        
        # Randomize start time (±2 seconds)
        source.start_time += np.random.uniform(-2.0, 2.0)
        source.start_time = max(0, source.start_time)
        
        # Randomize volume (±20%)
        volume_jitter = np.random.uniform(0.8, 1.2)
        source.volume_scale *= volume_jitter
    
    # Randomize noise level (±5 dB)
    config.background_noise_level += np.random.uniform(-5.0, 5.0)
    
    # Randomly adjust ambient levels
    for condition in config.ambient.values():
        condition.level_db += np.random.uniform(-3.0, 3.0)
    
    return config

def visualize_scene_setup(config: SimulationConfig, save_path: str = None):
    """Create a visualization of the acoustic scene setup"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Top-down view of scene
    ax1.set_aspect('equal')
    
    # Plot microphone array
    mic_x = [mic.x for mic in config.mic_positions]
    mic_y = [mic.y for mic in config.mic_positions]
    ax1.scatter(mic_x, mic_y, c='red', s=100, marker='s', label='Microphones', zorder=5)
    ax1.plot([0], [0], 'ro', markersize=8, label='Array Center')
    
    # Plot sources with different colors by type
    source_types = {}
    for i, source in enumerate(config.sources):
        x, y, z = source.position
        if source.source_type not in source_types:
            source_types[source.source_type] = len(source_types)
        
        color = plt.cm.Set1(source_types[source.source_type])
        ax1.scatter(x, y, c=[color], s=150, marker='o', 
                   label=f'{source.source_type}' if source.source_type not in [s.get_label() for s in ax1.get_children()],
                   alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add source labels
        ax1.annotate(f'{source.source_type}\n({x:.0f}, {y:.0f})', 
                    (x, y), xytext=(10, 10), textcoords='offset points',
                    fontsize=8, ha='left')
    
    # Add range circles
    for radius in [50, 100, 200, 500]:
        circle = plt.Circle((0, 0), radius, fill=False, linestyle='--', alpha=0.3, color='gray')
        ax1.add_patch(circle)
        ax1.text(radius*0.7, radius*0.7, f'{radius}m', fontsize=8, alpha=0.7)
    
    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Y Position (m)')
    ax1.set_title('Acoustic Scene Layout (Top View)')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Timeline view
    for i, source in enumerate(config.sources):
        start_time = source.start_time
        # Estimate duration (would need actual audio file info)
        estimated_duration = 3.0  # Default estimate
        
        ax2.barh(i, estimated_duration, left=start_time, 
                color=plt.cm.Set1(source_types[source.source_type]), 
                alpha=0.7, edgecolor='black')
        ax2.text(start_time + estimated_duration/2, i, 
                f'{source.source_type}', ha='center', va='center', fontsize=8)
    
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Source Index')
    ax2.set_title('Source Timeline')
    ax2.set_xlim(0, config.scene_duration)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Scene visualization saved to {save_path}")
    else:
        plt.show()

def analyze_simulation_results(mic_signals: np.ndarray, metadata: Dict):
    """Analyze and visualize simulation results"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Time domain signals for each microphone
    time = np.arange(mic_signals.shape[1]) / metadata['sample_rate']
    
    for i in range(4):
        axes[0, 0].plot(time, mic_signals[i], label=f'Mic {i+1}', alpha=0.7)
    
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].set_title('Microphone Signals')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Cross-correlation between mic pairs for TDOA visualization
    max_lag_samples = int(0.01 * metadata['sample_rate'])  # 10ms max lag
    
    # Example: Cross-correlation between mic 0 and mic 1
    correlation = np.correlate(mic_signals[1], mic_signals[0], mode='full')
    lags = np.arange(-len(mic_signals[0])+1, len(mic_signals[0]))
    
    # Focus on reasonable lag range
    center = len(lags) // 2
    lag_range = slice(center - max_lag_samples, center + max_lag_samples + 1)
    
    axes[0, 1].plot(lags[lag_range] / metadata['sample_rate'] * 1000, 
                   correlation[lag_range])
    axes[0, 1].set_xlabel('Lag (ms)')
    axes[0, 1].set_ylabel('Cross-correlation')
    axes[0, 1].set_title('Cross-correlation (Mic 1 vs Mic 0)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Power spectral density
    from scipy import signal as sp_signal
    
    freqs, psd = sp_signal.welch(mic_signals[0], metadata['sample_rate'], nperseg=512)
    axes[1, 0].semilogy(freqs, psd)
    axes[1, 0].set_xlabel('Frequency (Hz)')
    axes[1, 0].set_ylabel('PSD (V²/Hz)')
    axes[1, 0].set_title('Power Spectral Density (Mic 0)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Expected vs actual TDOA (if ground truth available)
    if 'ground_truth' in metadata and 'expected_tdoas' in metadata['ground_truth']:
        expected_tdoas = metadata['ground_truth']['expected_tdoas']
        if expected_tdoas:
            # Show expected TDOAs for first source
            tdoas = expected_tdoas[0]
            pairs = [f"Mic {t['mic_pair'][0]}-{t['mic_pair'][1]}" for t in tdoas]
            values = [t['tdoa_seconds'] * 1000 for t in tdoas]  # Convert to ms
            
            axes[1, 1].bar(pairs, values)
            axes[1, 1].set_xlabel('Microphone Pairs')
            axes[1, 1].set_ylabel('Expected TDOA (ms)')
            axes[1, 1].set_title('Expected Time Differences (First Source)')
            axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("Simulation Results Summary:")
    print(f"Duration: {len(mic_signals[0])/metadata['sample_rate']:.1f} seconds")
    print(f"Sample rate: {metadata['sample_rate']} Hz")
    print(f"Peak amplitude: {np.max(np.abs(mic_signals)):.3f}")
    print(f"RMS level: {np.sqrt(np.mean(mic_signals**2)):.3f}")
    
    if 'sources' in metadata:
        print(f"\nSources in scene:")
        for i, source in enumerate(metadata['sources']):
            pos = source['position']
            print(f"  {i+1}. {source.get('source_type', 'unknown')} at ({pos[0]:.0f}, {pos[1]:.0f}, {pos[2]:.0f})")

def validate_against_known_doa(mic_signals: np.ndarray, metadata: Dict, 
                              expected_azimuth: float, tolerance_deg: float = 10.0):
    """
    Simple validation function to check if DOA estimation would work
    This is a placeholder - you would replace with your actual DOA algorithm
    """
    
    # This is where you would call your actual ZODAS processing pipeline
    # For now, we'll just show the expected results
    
    if 'ground_truth' in metadata:
        gt = metadata['ground_truth']
        if 'source_directions' in gt and gt['source_directions']:
            estimated_azimuth = gt['source_directions'][0]['azimuth_deg']
            error = abs(estimated_azimuth - expected_azimuth)
            if error > 180:
                error = 360 - error  # Handle wraparound
            
            print(f"DOA Validation:")
            print(f"Expected azimuth: {expected_azimuth:.1f}°")
            print(f"Ground truth: {estimated_azimuth:.1f}°") 
            print(f"Error: {error:.1f}°")
            print(f"Within tolerance ({tolerance_deg}°): {'Yes' if error <= tolerance_deg else 'No'}")
            
            return error <= tolerance_deg
    
    return False

def run_example_simulation():
    """Run a complete example simulation with visualization"""
    
    print("Running example acoustic scene simulation...")
    
    # Create generator
    generator = AcousticSceneGenerator()
    
    # Use validation scenario (simple, clean)
    config = create_validation_scenario()
    
    print("Configuration:")
    print(f"- Duration: {config.scene_duration} seconds")
    print(f"- Sources: {len(config.sources)}")
    print(f"- Background noise: {config.background_noise_level} dB")
    
    # Visualize setup
    visualize_scene_setup(config)
    
    # This would require actual audio files to run
    print("\nTo run the complete simulation, you need:")
    print("1. Create a 'sounds/' directory")
    print("2. Add audio files referenced in the configuration")
    print("3. Run: mic_signals, metadata = generator.generate_scene(config)")
    print("4. Analyze results with: analyze_simulation_results(mic_signals, metadata)")
    
    return config

if __name__ == "__main__":
    # Run example
    example_config = run_example_simulation()
    
    print("\nExample configurations available:")
    print("- create_elephant_detection_scenario()")
    print("- create_multi_elephant_scenario()")  
    print("- create_noisy_environment_scenario()")
    print("- create_validation_scenario()")
    
    print("\nTo generate a full training dataset:")
    print("generate_training_dataset('output_directory', num_scenes=100)")
