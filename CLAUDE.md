# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

This project uses `uv` for dependency management:

- **Install dependencies**: `uv sync`  
- **Run tests**: `uv run pytest`
- **Format code**: `uv run black .`
- **Run demo**: `uv run python demo_script.py`
- **Run main application**: `uv run python main.py`
- **Run Streamlit app**: `uv run streamlit run app.py`

## Architecture

This is a wildlife acoustic simulation library built on pyroomacoustics for generating synthetic acoustic scenes for microphone array algorithm validation, specifically targeting elephant monitoring systems.

### Core Components

**Enhanced Pydantic Models (`models.py`)**:
- `SimulationConfig` - Complete configuration with validation
- `SoundSource`, `EnvironmentConfig`, `AmbientCondition` - Validated data models
- `SourceType`, `EnvironmentType` enums for type safety
- Helper functions: `create_elephant_source()`, `create_forest_environment()`

**Acoustic Scene Generation (`acoustic_scene_generator.py`)**:
- Uses pyroomacoustics for 3D acoustic propagation modeling
- Models ReSpeaker USB 4 Mic Array geometry precisely  
- Generates ground truth metadata for DOA/TDOA validation
- Handles realistic physics: inverse square law, atmospheric absorption, time delays

**Dataset Management (`dataset_manager.py`)**:
- `DatasetManager` - Handles Kaggle, Google Drive, URL, and local datasets
- `AudioCatalog` - Indexes and searches audio files by metadata
- `SmartAudioSelector` - Intelligently selects appropriate audio files
- Supports multiple audio formats and automatic cataloging

**Streamlit Web Interface (`app.py`)**:
- Interactive configuration builder with visual feedback
- Real-time simulation execution with progress tracking
- Analysis and visualization of results (signals, DOA, scene layout)
- Export functionality for audio, metadata, and configurations

**Hydra Configuration System (`hydraconf.py`)**:
- Structured configuration management with YAML files
- Predefined scenarios and environment presets
- Support for experiment sweeps and parameter studies
- Integration with Pydantic models for validation

**Configuration Builder (`config_builder.py`)**:
- `EnhancedSceneConfigBuilder` - Fluent API for scenario creation
- Preset scenarios: single elephant, multi-source, complex forest environments
- Supports interactive and programmatic configuration
- JSON serialization for scenario persistence

### Research Workflow

**1. Dataset Setup**:
```bash
# Add datasets via Streamlit interface or programmatically
uv run streamlit run app.py  # Use Dataset Manager tab
```

**2. Configuration Creation**:
```python
# Using Pydantic models
from enhanced_models import *
config = SimulationConfig(
    sources=[create_elephant_source(100, 45, "elephant.wav")],
    environment=create_forest_environment(),
    ambient={}
)

# Using fluent API
from config_builder import EnhancedSceneConfigBuilder
config = (EnhancedSceneConfigBuilder()
    .add_elephant(distance=100, azimuth_deg=45)
    .set_forest_clearing()
    .build())
```

**3. Hydra Experiments**:
```bash
# Run with configuration files
uv run python hydraconf.py --config-path=configs --config-name=single_elephant

# Create configuration templates
uv run python hydraconf.py create-configs
```

**4. Simulation Execution**:
```python
# Programmatic execution
from acoustic_scene_generator import WildlifeAcousticSimulator
simulator = WildlifeAcousticSimulator()
mic_signals, metadata = simulator.simulate_scene(config)

# Web interface execution via Streamlit
```

**5. Analysis & Validation**:
- DOA algorithm comparison (MUSIC, SRP-PHAT, FRIDA)
- Ground truth vs estimation error analysis  
- Signal analysis (time/frequency domain, spectrograms)
- Scene visualization (3D layout, range circles)

### Key Data Structures

- `SoundSource`: Audio file, 3D position, timing, volume, type metadata
- `EnvironmentConfig`: Room acoustics, absorption, temperature, humidity
- `MicrophonePosition`: 3D coordinates matching ReSpeaker array
- `AmbientCondition`: Rain, wind, forest background modeling
- `SimulationConfig`: Complete validated configuration container

### Signal Processing Pipeline

The system generates 4-channel microphone array outputs (16kHz) that match ZODAS processing pipeline requirements. Ground truth includes expected DOA angles, TDOA values, distances, and SNR information for algorithm validation.

### Hardware Integration

Specifically designed for ReSpeaker USB 4 Mic Array with precise geometry modeling: (-32,0,0), (0,-32,0), (32,0,0), (0,32,0) mm positioning at 16kHz sampling rate.

### Testing

Run integration tests to verify all components:
```bash
uv run python integration_test.py
```