# wildsim-audio
Python library for generating synthetic acoustic scenes for wildlife monitoring  and microphone array algorithm validation

This is a comprehensive acoustic scene simulation library for wildlife monitoring project.

### **Main Components:**

1. **`acoustic_scene_generator.py`** - Core simulation library
   - Handles 3D acoustic propagation physics
   - Manages multiple sound sources and ambient conditions  
   - Generates 4-channel outputs matching your ReSpeaker array
   - Produces detailed ground truth metadata

2. **`config_builder.py`** - Easy configuration interface
   - Interactive and programmatic scenario creation
   - Preset scenarios for common use cases
   - Save/load configurations to/from JSON

3. **`demo_script.py`** - Complete working demonstration
   - Generates synthetic audio so you can test immediately
   - Shows visualization and analysis of results
   - No external audio files needed to start

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

### **Usage Examples:**

```python
# Quick single elephant scene
config = quick_elephant_scene(distance=100, azimuth=45, noise_level=40)

# Interactive configuration
config = interactive_config_builder()

# Preset complex scenario  
config = create_preset_scenarios()['elephant_with_birds'].build()

# Generate scene
generator = AcousticSceneGenerator()
mic_signals, metadata = generator.generate_scene(config)

# Output: mic_signals is (4, N) array - exactly what your ZODAS pipeline expects
```
