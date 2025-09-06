# Forest Dataset Renaming Summary

## Overview
Successfully renamed 2,025 forest audio files based on their class metadata from the FSC22 dataset.

## Source Information
- **Original Dataset**: Forest Sound Classification (FSC22) 
- **Metadata File**: `Metadata V1.0 FSC22.csv`
- **Original Files Location**: `/Users/abhinav/antamaya/wildsim-audio/audio_cache/forest/Audio Wise V1.0-20220916T202003Z-001/Audio Wise V1.0/`
- **Renamed Files Location**: `/Users/abhinav/antamaya/wildsim-audio/audio_cache/forest/renamed_files/`

## Renaming Convention
Files were renamed following the pattern: `{class_name}{counter:02d}.wav`

Examples:
- `Fire` class: `fire01.wav`, `fire02.wav`, ..., `fire75.wav`
- `Rain` class: `rain01.wav`, `rain02.wav`, ..., `rain75.wav`
- `BirdChirping` class: `birdchirping01.wav`, `birdchirping02.wav`, ..., `birdchirping75.wav`

## Dataset Classes (27 classes, 75 files each)

### Natural Sounds
1. **Fire** - 75 files (fire01.wav - fire75.wav)
2. **Rain** - 75 files (rain01.wav - rain75.wav) 
3. **Thunderstorm** - 75 files (thunderstorm01.wav - thunderstorm75.wav)
4. **WaterDrops** - 75 files (waterdrops01.wav - waterdrops75.wav)
5. **Wind** - 75 files (wind01.wav - wind75.wav)

### Animal Sounds
6. **BirdChirping** - 75 files (birdchirping01.wav - birdchirping75.wav)
7. **Frog** - 75 files (frog01.wav - frog75.wav)
8. **Insect** - 75 files (insect01.wav - insect75.wav)
9. **Lion** - 75 files (lion01.wav - lion75.wav)
10. **WolfHowl** - 75 files (wolfhowl01.wav - wolfhowl75.wav)
11. **Squirrel** - 75 files (squirrel01.wav - squirrel75.wav)
12. **WingFlaping** - 75 files (wingflaping01.wav - wingflaping75.wav)

### Human Sounds
13. **Speaking** - 75 files (speaking01.wav - speaking75.wav)
14. **Footsteps** - 75 files (footsteps01.wav - footsteps75.wav)
15. **Clapping** - 75 files (clapping01.wav - clapping75.wav)
16. **Whistling** - 75 files (whistling01.wav - whistling75.wav)

### Tool/Equipment Sounds
17. **Axe** - 75 files (axe01.wav - axe75.wav)
18. **Chainsaw** - 75 files (chainsaw01.wav - chainsaw75.wav)
19. **Handsaw** - 75 files (handsaw01.wav - handsaw75.wav)
20. **WoodChop** - 75 files (woodchop01.wav - woodchop75.wav)
21. **Generator** - 75 files (generator01.wav - generator75.wav)

### Vehicle Sounds
22. **Helicopter** - 75 files (helicopter01.wav - helicopter75.wav)
23. **VehicleEngine** - 75 files (vehicleengine01.wav - vehicleengine75.wav)

### Explosive/Impact Sounds
24. **Gunshot** - 75 files (gunshot01.wav - gunshot75.wav)
25. **Firework** - 75 files (firework01.wav - firework75.wav)
26. **TreeFalling** - 75 files (treefalling01.wav - treefalling75.wav)

### Ambient
27. **Silence** - 75 files (silence01.wav - silence75.wav)

## Processing Results
- **Total Files Processed**: 2,025
- **Successfully Renamed**: 2,025
- **Errors**: 0
- **Processing Time**: < 1 minute

## Benefits of Renaming
1. **Intuitive naming**: File names clearly indicate their content class
2. **Organized structure**: Files are naturally sorted by class when listed alphabetically
3. **Easy identification**: No need to cross-reference metadata to understand file content
4. **Machine learning ready**: Perfect for training data with clear class labels
5. **User-friendly**: Audio engineers and researchers can quickly find relevant sounds

## Usage Example
To use fire sounds in your acoustic simulation:
```python
fire_sounds = [f"fire{i:02d}.wav" for i in range(1, 76)]  # fire01.wav to fire75.wav
```

## Original Mapping Preserved
The complete mapping from original numbered files to class-based names is documented in the processing script output and can be reconstructed from the metadata CSV if needed.

Date: September 6, 2025
Script: `forest_dataset_renamer.py`
