# Scene Generator Sound Selection Improvements

## Problem Fixed
The scene generator was previously ignoring user configurations for sound type selection and always using all available sound classes from the dataset. This made it impossible to create targeted acoustic scenes with specific sound categories.

## Solution Implemented

### 1. Enhanced GeneratorConfig
- Added `allowed_directional_types` and `allowed_ambient_types` parameters
- These allow users to specify exactly which sound types to use

### 2. Updated Sound Generation Logic
- Modified `_generate_directional_sounds()` to respect configured directional types
- Modified `_generate_ambient_sounds()` to respect configured ambient types
- Added validation to ensure configured types are available in the dataset

### 3. Command Line Interface
- Added `--allowed-directional` argument for specifying directional sound types
- Added `--allowed-ambient` argument for specifying ambient sound types
- Added `--list-types` argument to show all available sound types

### 4. Validation and Error Handling
- Automatic validation of configured sound types against available files
- Graceful fallback to defaults if invalid types are specified
- Clear feedback about which types are being used

## Usage Examples

### List Available Sound Types
```bash
python scene_generator.py --list-types
```

### Generate with Specific Sound Types
```bash
# Only bird and lion sounds for directional placement
python scene_generator.py --allowed-directional bird lion

# Only rain for ambient background
python scene_generator.py --allowed-ambient rain

# Combine both directional and ambient restrictions
python scene_generator.py --allowed-directional bird lion bear --allowed-ambient rain wind

# Full generation with restrictions
python scene_generator.py --num-samples 100 --allowed-directional bird lion --allowed-ambient rain --output-dir targeted_scenes
```

## Technical Implementation Details

### Sound Type Validation
- Builds cache of available sound types from dataset
- Validates configured types against available files
- Filters out invalid types with clear warnings
- Falls back to defaults if no valid types remain

### Elephant Handling
- Elephants are still handled separately for positive samples
- Elephant exclusion for negative samples is maintained
- No changes to elephant group generation logic

## Benefits

1. **Targeted Scene Creation**: Generate scenes with specific sound categories
2. **Dataset Flexibility**: Works with any dataset configuration
3. **Error Prevention**: Validates sound types before generation
4. **User Feedback**: Clear information about available and selected types
5. **Backward Compatibility**: Works without configuration (uses all types)

## Testing Verification

Generated test scenes with:
- Directional: bird, lion only
- Ambient: rain, wind only
- Result: 30 bird sounds, 39 lion sounds, 5 elephants (positive), 7 wind, 2 rain
- ✅ Confirmed working correctly - no other sound types used

## Files Modified

1. `scene_generator.py`:
   - Enhanced `GeneratorConfig` dataclass
   - Updated `_generate_directional_sounds()` method
   - Updated `_generate_ambient_sounds()` method
   - Added `_validate_sound_type_config()` method
   - Enhanced command line arguments
   - Added `--list-types` functionality
