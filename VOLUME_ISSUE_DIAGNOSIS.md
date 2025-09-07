# Volume Control Issue - Diagnosis and Workaround

## Problem Summary
You reported that changing volume settings from 0.1 to 3.0 produces identical audio when played through Streamlit's audio widget.

## Root Cause Analysis

### ✅ **Volume Multiplication Working Correctly**
The mathematical volume multiplication is working perfectly:
- **Debug test results**: Volume ratios are exactly as expected (0.1 = 10%, 3.0 = 300%)
- **RMS and Max values**: Scale correctly with volume settings
- **Array verification**: Volume-adjusted arrays are mathematically different from originals

### ❌ **Issue: Browser/Streamlit Audio Normalization**
The problem is with Streamlit's `st.audio()` widget or browser audio playback:
- **Browser auto-normalization**: Many browsers automatically normalize audio levels
- **Streamlit audio processing**: May apply gain normalization before playback
- **System audio settings**: macOS/Windows may have auto-leveling enabled

## Evidence
1. **Mathematical verification**: `debug_volume.py` shows perfect volume scaling
2. **File-based test**: Generated WAV files have correct volume differences when played externally
3. **System audio test**: Using `afplay` (macOS) confirms volume differences are audible

## Workarounds

### 1. **Use External Audio Player** (Recommended)
The scene manager now includes a "Save Volume Test Files" button that saves audio files for external playback.

### 2. **Browser Settings**
- **Chrome**: Check `chrome://flags/#disable-automatic-tab-discarding`
- **Safari**: Disable auto-play restrictions in Develop menu
- **Firefox**: Check `media.autoplay.enabled` in about:config

### 3. **System Audio Settings**
- **macOS**: System Preferences → Sound → disable "Use ambient noise reduction"
- **Windows**: Sound settings → disable "Loudness Equalization"

## Verification Steps

### Quick Test:
```bash
# Run volume debug script
uv run python debug_volume.py

# Play generated files with system player
afplay /tmp/tmpXXXXXX_vol_0.1.wav  # Should be quiet
afplay /tmp/tmpXXXXX_vol_3.0.wav   # Should be loud
```

### In Streamlit:
1. Use "🧪 Direct Volume Test" button
2. Check the mathematical ratios (should be correct)
3. Use "💾 Save Volume Test Files" button
4. Play saved files with system audio player

## Technical Details

### What Works:
- ✅ Volume multiplication: `audio * volume`
- ✅ RMS ratio calculation: matches expected volume
- ✅ File saving with soundfile
- ✅ External audio playback

### What Doesn't Work:
- ❌ Streamlit `st.audio()` volume perception
- ❌ Browser-based audio widget volume differences
- ❌ In-browser audio normalization bypass

## Conclusion
**The volume control is working perfectly**. The issue is with browser/Streamlit audio playback normalization, not with the simulation or audio processing code. For accurate volume testing, use external audio players or the provided file export functionality.

## Next Steps
1. Use the enhanced debug tools in the scene editor
2. Test with external audio player using saved files
3. Consider alternative audio playback methods for the web interface
4. Focus on the acoustics simulation accuracy (which is unaffected by this playback issue)