# Modular Code Structure

This document explains the modular organization of the Wildlife Acoustic Simulation codebase.

## Core Modules

### 1. `app.py` - Main Streamlit Application
- **Purpose**: Main web interface with integrated tabs
- **Navigation**: 
  - Dataset Manager (Add/Browse/Manage datasets)
  - Scene Creator (New integrated tab)
  - Config Builder (Legacy configuration)
  - Analysis Dashboard (Results analysis)
- **Dependencies**: All other modules

### 2. `scene_creator.py` - Modular Scene Creator Component
- **Purpose**: Encapsulates all scene creation UI logic
- **Class**: `SceneCreatorUI` - Self-contained scene creation interface
- **Function**: `render_scene_creator_tab()` - Main integration point
- **Features**:
  - Sound placement in 3D space
  - File selection from datasets 
  - 3D scene visualization
  - Simulation execution

### 3. `simple_config.py` - Simplified Configuration System
- **Purpose**: Clean dataclasses for scene configuration
- **Classes**:
  - `SoundPlacement` - 3D positioned audio sources
  - `AmbientSound` - Background/ambient conditions
  - `SceneConfig` - Complete scene configuration
  - `SceneBuilder` - Helper for building scenes

### 4. `simple_simulator.py` - Clean Simulation Engine
- **Purpose**: Streamlined simulation using real audio data
- **Class**: `SimpleSimulator` - Main simulation engine
- **Features**:
  - Uses pyroomacoustics for acoustic modeling
  - ReSpeaker 4-mic array geometry
  - Real audio file integration
  - Ground truth generation

### 5. `dataset_manager.py` - Dataset Management
- **Purpose**: Handles audio dataset cataloging and access
- **Classes**:
  - `DatasetManager` - Main dataset interface
  - `AudioCatalog` - File indexing and search
  - `DatasetDownloader` - Multi-source downloading

## Integration Architecture

```
app.py (Main Interface)
├── Dataset Manager Page
│   ├── Add/Browse datasets
│   └── File cataloging
├── Scene Creator Page ← PRIMARY SIMULATOR
│   ├── scene_creator.py (UI and workflow)
│   ├── simple_config.py (Configuration)
│   └── simple_simulator.py (Simulation engine)
├── Analysis Page
├── Export/Import Page
└── Help Page
```

## Key Benefits of Modular Design

### 1. **Separation of Concerns**
- UI logic separated from business logic
- Scene creation is self-contained
- Easy to test individual components

### 2. **Maintainability**
- Each module has a clear responsibility
- Changes to scene creator don't affect other parts
- Easy to add new features

### 3. **Reusability**
- Scene creator can be used in other contexts
- Simulation engine is standalone
- Configuration system is framework-agnostic

### 4. **Code Organization**
- Main app.py stays manageable (~1800 lines vs potential 3000+)
- Related functionality grouped together
- Clear import dependencies

## Usage Examples

### Using Scene Creator Directly
```python
from scene_creator import SceneCreatorUI
from dataset_manager import DatasetManager

dm = DatasetManager()
scene_creator = SceneCreatorUI(dm)
# Use in custom Streamlit app
```

### Using Simulation Engine Directly
```python
from simple_simulator import SimpleSimulator
from simple_config import SceneConfig

sim = SimpleSimulator(dataset_manager)
mic_signals, metadata = sim.simulate_scene(config)
```

### Integrating in Main App
```python
from scene_creator import render_scene_creator_tab

# In main app.py scene creator page
def scene_creator_page():
    dataset_manager = st.session_state.dataset_manager
    render_scene_creator_tab(dataset_manager)
```

## File Organization

```
├── app.py                    # Main Streamlit interface
├── scene_creator.py          # Scene creator UI module  
├── simple_config.py          # Configuration dataclasses
├── simple_simulator.py       # Simulation engine
├── dataset_manager.py        # Dataset management
├── auth_helpers.py           # Authentication utilities
├── configs/                  # YAML configuration files
│   ├── simple_elephant.yaml
│   └── multi_elephant.yaml
└── outputs/                  # Simulation results
```

## Migration Notes

- `simple_app.py` → Moved to `simple_app_backup.py`
- Functionality fully integrated into main `app.py`
- Scene Creator is now the **primary simulator** - replacing separate Configuration and Simulation pages
- Single application entry point at http://localhost:8504
- Scene Creator provides complete workflow: configuration → simulation → results
- Removed redundant Configuration and Simulation pages for streamlined experience

## Development Workflow

1. **Adding Scene Creator Features**: Edit `scene_creator.py`
2. **Modifying Simulation Logic**: Edit `simple_simulator.py`
3. **Changing Configuration**: Edit `simple_config.py`
4. **Adding Dataset Features**: Edit `dataset_manager.py`
5. **Main App Changes**: Edit `app.py` (minimal changes needed)

This modular structure makes the codebase more maintainable while keeping all functionality accessible through a single, integrated interface.