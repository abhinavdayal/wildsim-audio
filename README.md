# wildsim-audio
Python library for generating synthetic acoustic scenes for wildlife monitoring and microphone array algorithm validation

This is a comprehensive acoustic scene simulation library built on pyroomacoustics for generating synthetic acoustic scenes specifically targeting elephant monitoring systems and microphone array algorithm validation.

## Installation

This project uses `uv` for dependency management:

```bash
# Install dependencies
uv sync

# Run tests  
uv run pytest

# Format code
uv run black .
```

## Core Components

### **Enhanced Pydantic Models (`models.py`)**
- `SimulationConfig` - Complete configuration with validation and type safety
- `SoundSource`, `EnvironmentConfig`, `AmbientCondition` - Validated data models
- `SourceType`, `EnvironmentType` enums for type safety
- Helper functions: `create_elephant_source()`, `create_forest_environment()`

### **Acoustic Scene Generation (`acoustic_scene_generator.py`)**
- Uses pyroomacoustics for 3D acoustic propagation modeling
- Models ReSpeaker USB 4 Mic Array geometry precisely  
- Generates ground truth metadata for DOA/TDOA validation
- Handles realistic physics: inverse square law, atmospheric absorption, time delays

### **Dataset Management (`dataset_manager.py`)**
- `DatasetManager` - Handles Kaggle, Google Drive, URL, and local datasets
- `AudioCatalog` - Indexes and searches audio files by metadata
- `SmartAudioSelector` - Intelligently selects appropriate audio files
- Supports multiple audio formats and automatic cataloging

### **Streamlit Web Interface (`app.py`)**
- Interactive configuration builder with visual feedback
- Real-time simulation execution with progress tracking
- Analysis and visualization of results (signals, DOA, scene layout)
- Export functionality for audio, metadata, and configurations

### **Hydra Configuration System (`hydraconf.py`)**
- Structured configuration management with YAML files
- Predefined scenarios and environment presets
- Support for experiment sweeps and parameter studies
- Integration with Pydantic models for validation

### **Configuration Builder (`config_builder.py`)**
- `EnhancedSceneConfigBuilder` - Fluent API for scenario creation
- Preset scenarios: single elephant, multi-source, complex forest environments
- Supports interactive and programmatic configuration
- JSON serialization for scenario persistence

### **Key Features:**

**Exact Hardware Matching:**
- ReSpeaker USB 4 Mic Array geometry: (-32,0,0), (0,-32,0), (32,0,0), (0,32,0) mm
- 16kHz sampling rate, proper time delays and attenuation
- Matches your ZODAS processing pipeline requirements

**Realistic Physics:**
- 3D acoustic propagation with inverse square law attenuation
- Time-of-arrival delays calculated precisely for each microphone
- Frequency-dependent atmospheric absorption for distant sources

**Comprehensive Scenarios:**
- Single elephant detection (clean validation)
- Multi-elephant herds (source separation testing)
- Noisy environments (detection limit testing)
- Complex forest soundscapes (real-world conditions)

**Ground Truth Generation:**
- Expected DOA angles for each source
- Precise TDOA values between all microphone pairs
- Distance and SNR information
- Perfect validation data for your algorithms

## Quick Start

### 1. Web Interface (Recommended)
```bash
# Launch the Streamlit web interface
uv run streamlit run app.py
```
Use the web interface to:
- Set up datasets via the Dataset Manager tab
- Create configurations interactively 
- Run simulations with real-time progress
- Visualize and analyze results

### 2. Programmatic Usage

#### Using Pydantic Models (Type-Safe)
```python
from models import *

# Create configuration with validation
config = SimulationConfig(
    sources=[create_elephant_source(100, 45, "elephant.wav")],
    environment=create_forest_environment(),
    ambient=AmbientCondition()
)

# Run simulation
from acoustic_scene_generator import WildlifeAcousticSimulator
simulator = WildlifeAcousticSimulator()
mic_signals, metadata = simulator.simulate_scene(config)
```

#### Using Fluent Configuration Builder
```python
from config_builder import EnhancedSceneConfigBuilder

# Build complex scenarios fluently
config = (EnhancedSceneConfigBuilder()
    .add_elephant(distance=100, azimuth_deg=45)
    .add_elephant(distance=200, azimuth_deg=-30) 
    .set_forest_clearing()
    .add_ambient_rain(intensity=0.3)
    .build())

# Generate scene
simulator = WildlifeAcousticSimulator()
mic_signals, metadata = simulator.simulate_scene(config)
```

#### Using Hydra for Experiments
```bash
# Run with configuration files
uv run python hydraconf.py --config-path=configs --config-name=single_elephant

# Create configuration templates
uv run python hydraconf.py create-configs
```

### 3. Demo and Examples
```bash
# Run complete demonstration (no external files needed)
uv run python demo_script.py

# Run integration tests
uv run python integration_test.py
```

## Research Workflow

1. **Dataset Setup**: Use Streamlit interface or `DatasetManager` class to add audio datasets
2. **Configuration**: Create scenarios using Pydantic models, fluent builder, or Hydra configs
3. **Simulation**: Execute via web interface or programmatically
4. **Analysis**: Compare DOA algorithms (MUSIC, SRP-PHAT, FRIDA) against ground truth
5. **Validation**: Analyze estimation errors, signal quality, and scene visualization

## Dependencies

Built on modern Python scientific stack:
- **pyroomacoustics** - 3D acoustic simulation engine
- **pydantic** - Type-safe configuration validation
- **streamlit** - Interactive web interface
- **hydra-core** - Advanced configuration management
- **librosa** - Audio processing and analysis
- **numpy/scipy** - Numerical computing
- **matplotlib/plotly** - Visualization and analysis
