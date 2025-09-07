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

### **JSON Configuration System**
- Scene configuration management with JSON files
- Predefined scenarios and environment presets
- Interactive configuration via Streamlit interface
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

## Sound Timing and Placement System

The simulation system handles temporal placement of sounds within scenes using a sophisticated timing mechanism:

### Start Time Selection
- **Directional animal sounds**: Randomly placed between 0 and `(scene_duration - 5)` seconds
- **Elephant sounds**: Randomly placed between 0 and `(scene_duration - 8)` seconds  
- This ensures sounds have sufficient time to complete before scene ends

### Audio Duration Handling
- Individual animal sounds maintain their **natural duration** (typically 2-10 seconds)
- Sounds are **not** artificially stretched to fill remaining scene time
- If a 5-second bird call starts at second 20 in a 30-second scene, it ends naturally at second 25

### Multiple Sound Overlap
- Multiple sounds can overlap naturally (e.g., bird call at 5s, elephant at 7s)
- Each sound maintains independent acoustic characteristics
- Pyroomacoustics handles realistic acoustic propagation and summation at each microphone

### Example Timeline (30-second scene)
```
0-30s: Ambient forest sounds (continuous, looped)
2-7s:  Bird call (45° azimuth, 80m distance)
12-18s: Elephant trumpet (120° azimuth, 150m distance)  
25-28s: Monkey call (270° azimuth, 60m distance)
```

### Technical Implementation
The system uses pyroomacoustics' `delay` parameter for precise temporal control:
```python
room.add_source(position, signal=audio, delay=start_time)
```

This creates naturalistic acoustic scenes where discrete animal vocalizations occur sporadically within continuous ambient soundscapes, exactly as heard in real forest environments.

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

#### Using Streamlit Interface
```bash
# Run interactive scene creator and simulator
uv run streamlit run app.py
# Navigate to Scene Creator tab for interactive configuration and simulation
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
2. **Configuration**: Create scenarios using Pydantic models, fluent builder, or Streamlit interface
3. **Simulation**: Execute via web interface or programmatically
4. **Analysis**: Compare DOA algorithms (MUSIC, SRP-PHAT, FRIDA) against ground truth
5. **Validation**: Analyze estimation errors, signal quality, and scene visualization

## Dependencies

Built on modern Python scientific stack:
- **pyroomacoustics** - 3D acoustic simulation engine
- **pydantic** - Type-safe configuration validation
- **streamlit** - Interactive web interface
- **json** - Configuration file management
- **librosa** - Audio processing and analysis
- **numpy/scipy** - Numerical computing
- **matplotlib/plotly** - Visualization and analysis
