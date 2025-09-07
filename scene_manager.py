"""
Unified Scene Management System for Wildlife Acoustic Simulation
Handles scene editing, visualization, audio playback, and re-rendering for both single and batch workflows
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import soundfile as sf
import tempfile
import json
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union
import librosa

from simple_config import SceneConfig, SoundPlacement, AmbientSound
from simple_simulator import SimpleSimulator
from dataset_manager import DatasetManager


class SceneManager:
    """Unified scene management for both creator and generator workflows"""

    def __init__(self, dataset_manager: DatasetManager):
        self.dataset_manager = dataset_manager
        self.simulator = SimpleSimulator(dataset_manager)

    def render_scene_editor(
        self, scene: SceneConfig, scene_key: str = "default"
    ) -> SceneConfig:
        """
        Unified scene editor that can be used in both single scene creation and batch editing

        Args:
            scene: The scene to edit
            scene_key: Unique key for session state management

        Returns:
            Updated scene configuration
        """
        st.subheader(f"🎬 Scene: {scene.name}")

        # Create columns for layout
        col1, col2 = st.columns([2, 1])

        with col1:
            # Basic scene properties
            scene.name = st.text_input(
                "Scene Name", scene.name, key=f"name_{scene_key}"
            )
            scene.duration = st.slider(
                "Duration (seconds)",
                1.0,
                120.0,
                scene.duration,
                key=f"duration_{scene_key}",
            )
            scene.description = st.text_area(
                "Description", scene.description, key=f"description_{scene_key}"
            )

        with col2:
            # Quick actions
            if st.button(f"🎵 Simulate Scene", key=f"simulate_{scene_key}"):
                self._simulate_and_cache_scene(scene, scene_key)

            if st.button(f"📊 3D Visualization", key=f"visualize_{scene_key}"):
                self._show_scene_visualization(scene, scene_key)

        # Tabbed interface for different editing aspects
        tab1, tab2, tab3, tab4 = st.tabs(
            [
                "🎯 Directional Sounds",
                "🌲 Ambient Sounds",
                "🔊 Audio Playback",
                "📈 Analysis",
            ]
        )

        with tab1:
            scene = self._edit_directional_sounds(scene, scene_key)

        with tab2:
            scene = self._edit_ambient_sounds(scene, scene_key)

        with tab3:
            self._render_audio_playback(scene, scene_key)

        with tab4:
            self._render_scene_analysis(scene, scene_key)

        return scene

    def _edit_directional_sounds(
        self, scene: SceneConfig, scene_key: str
    ) -> SceneConfig:
        """Edit directional sounds with individual playback and isolation"""

        # Add new sound
        st.subheader("➕ Add Directional Sound")
        with st.expander("Add New Sound"):
            col1, col2, col3 = st.columns(3)

            with col1:
                sound_type = st.selectbox(
                    "Sound Type",
                    [
                        "elephant",
                        "bird",
                        "monkey",
                        "lion",
                        "bear",
                        "wolf",
                        "dog",
                        "cat",
                        "chicken",
                        "horse",
                        "cow",
                        "sheep",
                    ],
                    key=f"new_sound_type_{scene_key}",
                )
                distance = st.slider(
                    "Distance (m)", 10.0, 500.0, 100.0, key=f"new_distance_{scene_key}"
                )

            with col2:
                azimuth = st.slider(
                    "Azimuth (degrees)",
                    0.0,
                    360.0,
                    0.0,
                    key=f"new_azimuth_{scene_key}",
                )
                elevation = st.slider(
                    "Elevation (degrees)",
                    0.0,
                    90.0,
                    0.0,
                    key=f"new_elevation_{scene_key}",
                )

            with col3:
                start_time = st.slider(
                    "Start Time (s)",
                    0.0,
                    scene.duration,
                    0.0,
                    0.1,
                    key=f"new_start_time_{scene_key}",
                )
                volume = st.slider(
                    "Volume", 0.1, 3.0, 1.0, key=f"new_volume_{scene_key}"
                )

            # Audio file selection
            audio_file, _ = self._render_audio_file_selector(f"new_audio_{scene_key}")

            if st.button("Add Sound", key=f"add_sound_{scene_key}"):
                if audio_file:
                    new_sound = self._create_sound_placement(
                        audio_file,
                        sound_type,
                        distance,
                        azimuth,
                        elevation,
                        start_time,
                        volume,
                    )
                    scene.sounds.append(new_sound)
                    st.success(f"Added {sound_type} sound")
                    st.rerun()

        # Edit existing sounds
        st.subheader("🎯 Current Directional sounds")
        if scene.sounds:
            sounds_data = []
            for i, sound in enumerate(scene.sounds):
                sounds_data.append(
                    {
                        "Index": i,
                        "Type": sound.sound_type,
                        "File": (
                            Path(sound.audio_file).name if sound.audio_file else "None"
                        ),
                        "Distance": f"{sound.distance:.1f}m",
                        "Azimuth": f"{sound.azimuth:.1f}°",
                        "Elevation": f"{sound.elevation:.1f}°",
                        "Start": f"{sound.start_time:.1f}s",
                        "Volume": f"{sound.volume:.2f}",
                    }
                )

            df = pd.DataFrame(sounds_data)
            st.dataframe(df, use_container_width=True)

            # Individual sound editing
            selected_sound_idx = st.selectbox(
                "Select sound to edit/play",
                range(len(scene.sounds)),
                format_func=lambda x: f"{scene.sounds[x].sound_type} at {scene.sounds[x].distance:.1f}m",
                key=f"select_sound_{scene_key}",
            )

            if selected_sound_idx is not None:
                self._edit_individual_sound(scene, selected_sound_idx, scene_key)
        else:
            st.info("No directional sounds added yet.")

        return scene

    def _edit_individual_sound(
        self, scene: SceneConfig, sound_idx: int, scene_key: str
    ):
        """Edit and preview individual sound"""
        sound = scene.sounds[sound_idx]

        col1, col2 = st.columns([3, 1])

        with col1:
            st.subheader(f"🔧 Edit {sound.sound_type.title()} Sound")

            # Create sub-columns for parameters
            subcol1, subcol2, subcol3 = st.columns(3)

            with subcol1:
                new_distance = st.slider(
                    "Distance (m)",
                    10.0,
                    500.0,
                    sound.distance,
                    key=f"edit_distance_{scene_key}_{sound_idx}",
                )
                new_volume = st.slider(
                    "Volume",
                    0.1,
                    3.0,
                    sound.volume,
                    key=f"edit_volume_{scene_key}_{sound_idx}",
                )

            with subcol2:
                new_azimuth = st.slider(
                    "Azimuth (°)",
                    0.0,
                    360.0,
                    sound.azimuth if sound.azimuth >= 0 else sound.azimuth + 360,
                    key=f"edit_azimuth_{scene_key}_{sound_idx}",
                )
                new_elevation = st.slider(
                    "Elevation (°)",
                    0.0,
                    90.0,
                    abs(sound.elevation),
                    key=f"edit_elevation_{scene_key}_{sound_idx}",
                )

            with subcol3:
                new_start_time = st.slider(
                    "Start Time (s)",
                    0.0,
                    scene.duration,
                    sound.start_time,
                    0.1,
                    key=f"edit_start_time_{scene_key}_{sound_idx}",
                )
                
            # Update sound properties and recalculate position if changed
            position_changed = (
                new_distance != sound.distance or 
                new_azimuth != sound.azimuth or 
                new_elevation != sound.elevation
            )
            
            # Apply updates
            sound.distance = new_distance
            sound.volume = new_volume
            sound.azimuth = new_azimuth
            sound.elevation = new_elevation
            sound.start_time = new_start_time
            
            # Recalculate 3D position if spatial parameters changed
            if position_changed:
                azimuth_rad = np.radians(sound.azimuth)
                elevation_rad = np.radians(sound.elevation)
                
                x = sound.distance * np.cos(elevation_rad) * np.cos(azimuth_rad)
                y = sound.distance * np.cos(elevation_rad) * np.sin(azimuth_rad)
                z = sound.distance * np.sin(elevation_rad)
                
                sound.position = [x, y, z]
                
                st.caption(f"🔄 Position updated: ({x:.1f}, {y:.1f}, {z:.1f})m")
            
            # Show current parameter summary
            st.info(f"**Current Settings:**\nVol: {sound.volume:.2f}, Dist: {sound.distance:.1f}m\nAzimuth: {sound.azimuth:.1f}°, Start: {sound.start_time:.1f}s")

        with col2:
            # Individual sound actions
            if st.button(
                "🎵 Play Original", key=f"play_original_{scene_key}_{sound_idx}"
            ):
                if sound.audio_file and Path(sound.audio_file).exists():
                    st.write("🎧 **Original Raw Audio File** (no spatial effects):")
                    st.audio(sound.audio_file, format="audio/wav")
                    
                    # Show file info
                    try:
                        import librosa
                        audio_data, sr = librosa.load(sound.audio_file, sr=None)
                        duration = len(audio_data) / sr
                        max_amp = np.max(np.abs(audio_data))
                        rms = np.sqrt(np.mean(audio_data ** 2))
                        st.caption(f"File Info: Duration={duration:.1f}s, Max={max_amp:.3f}, RMS={rms:.4f}, SR={sr}Hz")
                    except Exception as e:
                        st.caption(f"File: {Path(sound.audio_file).name}")
                else:
                    st.error("Audio file not found")

            if st.button(
                "🎯 Play Isolated", key=f"play_isolated_{scene_key}_{sound_idx}"
            ):
                self._play_isolated_sound(sound, scene, scene_key, sound_idx)

            if st.button("🗑️ Remove", key=f"remove_{scene_key}_{sound_idx}"):
                scene.sounds.pop(sound_idx)
                st.success("Sound removed")
                st.rerun()
            
            # Direct volume test (bypass all simulation)
            if st.button("🧪 Direct Volume Test", key=f"direct_test_{scene_key}_{sound_idx}"):
                if sound.audio_file and Path(sound.audio_file).exists():
                    try:
                        import librosa
                        import numpy as np
                        
                        # Load original audio
                        audio_data, sr = librosa.load(sound.audio_file, sr=16000, duration=5.0)
                        
                        # Apply volume directly
                        volume_adjusted = audio_data * sound.volume
                        
                        st.write(f"🧪 **Direct Volume Test**: Original × {sound.volume:.3f}")
                        
                        # Detailed analysis
                        orig_rms = np.sqrt(np.mean(audio_data**2))
                        adj_rms = np.sqrt(np.mean(volume_adjusted**2))
                        orig_max = np.max(np.abs(audio_data))
                        adj_max = np.max(np.abs(volume_adjusted))
                        
                        st.write(f"- **Original**: RMS={orig_rms:.6f}, Max={orig_max:.6f}")
                        st.write(f"- **Volume-adjusted**: RMS={adj_rms:.6f}, Max={adj_max:.6f}")
                        st.write(f"- **RMS Ratio**: {(adj_rms / (orig_rms + 1e-10)):.6f} (should be {sound.volume:.3f})")
                        st.write(f"- **Max Ratio**: {(adj_max / (orig_max + 1e-10)):.6f} (should be {sound.volume:.3f})")
                        
                        # Show array stats
                        st.write(f"- **Original range**: [{np.min(audio_data):.6f}, {np.max(audio_data):.6f}]")
                        st.write(f"- **Adjusted range**: [{np.min(volume_adjusted):.6f}, {np.max(volume_adjusted):.6f}]")
                        
                        # Critical test: Are the arrays actually different?
                        if np.array_equal(audio_data, volume_adjusted):
                            st.error("🚨 **CRITICAL**: Arrays are identical! Volume multiplication failed!")
                        elif np.allclose(audio_data, volume_adjusted):
                            st.error("🚨 **CRITICAL**: Arrays are nearly identical! Volume multiplication barely working!")
                        else:
                            st.success("✅ Arrays are different - volume multiplication working mathematically")
                            
                        # Show first few samples as proof
                        st.write(f"**Sample values (first 5):**")
                        st.write(f"- Original: {audio_data[:5]}")
                        st.write(f"- Adjusted: {volume_adjusted[:5]}")
                        
                        # Side-by-side comparison
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**🎵 Original Audio:**")
                            st.audio(audio_data, format="audio/wav", sample_rate=sr)
                        
                        with col2:
                            st.write(f"**🔊 Volume × {sound.volume:.1f}:**")
                            st.audio(volume_adjusted, format="audio/wav", sample_rate=sr)
                        
                        # Additional extreme volume tests for troubleshooting
                        st.write("---")
                        st.write("**🔬 Extreme Volume Tests (for debugging only):**")
                        
                        # Test with very different volumes
                        very_quiet = audio_data * 0.01  # 1% volume  
                        very_loud = audio_data * 5.0    # 500% volume (may clip)
                        
                        col3, col4, col5 = st.columns(3)
                        with col3:
                            st.write("**Extreme Quiet (1%):**")
                            st.audio(very_quiet, format="audio/wav", sample_rate=sr)
                            
                        with col4:
                            st.write("**🎵 Original (100%):**")
                            st.audio(audio_data, format="audio/wav", sample_rate=sr)
                            
                        with col5:
                            st.write("**🔥 Extreme Loud (500%):**")
                            st.audio(very_loud, format="audio/wav", sample_rate=sr)
                        
                        # Save files temporarily for comparison
                        st.write("---")
                        st.write("**📁 Audio File Analysis (check browser developer tools):**")
                        
                        # Check if browser is normalizing audio
                        import tempfile
                        import soundfile as sf
                        
                        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f1:
                            sf.write(f1.name, audio_data, sr)
                            st.write(f"Original file size: {Path(f1.name).stat().st_size} bytes")
                            
                        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f2:
                            sf.write(f2.name, volume_adjusted, sr)
                            st.write(f"Volume-adjusted file size: {Path(f2.name).stat().st_size} bytes")
                        
                        # Browser audio normalization test
                        st.info("🔍 **If all volumes sound identical, your browser or audio system might be auto-normalizing audio levels. Try with different browsers or check system audio settings.**")
                        
                        st.warning("⚠️ **Listen Test**: Play both audio files above. If volume multiplication is working, the adjusted volume should sound significantly different!")
                        
                        # File-based playback alternative
                        st.write("---")
                        st.write("**💾 Alternative: Save Files for External Playback**")
                        st.info("If Streamlit audio sounds identical, save files and play with your system's audio player")
                        
                        save_button_key = f"save_files_{scene_key}_{sound_idx}"
                        if st.button("💾 Save Volume Test Files", key=save_button_key):
                            st.session_state[f"saving_{save_button_key}"] = True
                            
                        if st.session_state.get(f"saving_{save_button_key}", False):
                            st.write("🔄 Button clicked - starting save process...")
                            
                            # Reset the button state
                            st.session_state[f"saving_{save_button_key}"] = False
                            
                            try:
                                import soundfile as sf
                                import os
                                
                                st.write("📁 Creating output directory...")
                                # Create output directory in project
                                current_dir = Path.cwd()
                                output_dir = current_dir / "debug_audio_output"
                                output_dir.mkdir(exist_ok=True)
                                
                                st.write(f"📂 Working directory: {current_dir}")
                                st.write(f"📂 Output directory: {output_dir}")
                                st.write(f"📂 Directory exists: {output_dir.exists()}")
                                
                                # Save files with descriptive names
                                base_name = f"volume_test_{Path(sound.audio_file).stem}"
                                
                                orig_file = output_dir / f"{base_name}_original.wav"
                                vol_file = output_dir / f"{base_name}_vol_{sound.volume:.1f}.wav"
                                
                                st.write(f"💾 Saving files:")
                                st.write(f"  - Original: {orig_file}")
                                st.write(f"  - Volume: {vol_file}")
                                
                                # Save the files
                                st.write("💾 Writing original file...")
                                sf.write(str(orig_file), audio_data, sr)
                                
                                st.write("💾 Writing volume-adjusted file...")
                                sf.write(str(vol_file), volume_adjusted, sr)
                                
                                st.write("🔍 Checking if files exist...")
                                # Verify files were created
                                if orig_file.exists() and vol_file.exists():
                                    orig_size = orig_file.stat().st_size
                                    vol_size = vol_file.stat().st_size
                                    
                                    st.success(f"✅ Files saved successfully!")
                                    st.write(f"📁 **Location**: `{output_dir.absolute()}`")
                                    st.write(f"- **Original**: `{orig_file.name}` ({orig_size:,} bytes)")
                                    st.write(f"- **Volume {sound.volume:.1f}**: `{vol_file.name}` ({vol_size:,} bytes)")
                                    
                                    # Show actual paths for command line
                                    st.write("**🎵 Command to play files:**")
                                    st.code(f"afplay '{orig_file.absolute()}'")
                                    st.code(f"afplay '{vol_file.absolute()}'")
                                    
                                else:
                                    st.error(f"❌ Files were not created!")
                                    st.write(f"Original exists: {orig_file.exists()}")
                                    st.write(f"Volume exists: {vol_file.exists()}")
                                    if output_dir.exists():
                                        files_in_dir = list(output_dir.iterdir())
                                        st.write(f"Files in output dir: {files_in_dir}")
                                    
                            except Exception as e:
                                st.error(f"❌ Save failed: {str(e)}")
                                import traceback
                                st.write("**Debug info:**")
                                st.code(traceback.format_exc())
                        
                    except Exception as e:
                        st.error(f"Direct test failed: {e}")
                else:
                    st.error("Cannot test - audio file not found")

    def _edit_ambient_sounds(self, scene: SceneConfig, scene_key: str) -> SceneConfig:
        """Edit ambient sounds"""
        st.subheader("🌲 Ambient Environment")

        # Add new ambient sound
        with st.expander("Add Ambient Sound"):
            col1, col2 = st.columns(2)

            with col1:
                ambient_type = st.selectbox(
                    "Ambient Type",
                    ["forest", "rain", "wind", "river", "night"],
                    key=f"new_ambient_type_{scene_key}",
                )
                level_db = st.slider(
                    "Level (dB)",
                    -60.0,
                    -20.0,
                    -40.0,
                    key=f"new_ambient_level_{scene_key}",
                )

            with col2:
                ambient_audio_file, _ = self._render_audio_file_selector(
                    f"new_ambient_{scene_key}"
                )

            if st.button("Add Ambient", key=f"add_ambient_{scene_key}"):
                new_ambient = AmbientSound(ambient_type, level_db, ambient_audio_file)
                scene.ambient.append(new_ambient)
                st.success(f"Added {ambient_type} ambient")
                st.rerun()

        # Edit existing ambient sounds
        if scene.ambient:
            for i, ambient in enumerate(scene.ambient):
                with st.expander(f"{ambient.sound_type.title()} Ambient"):
                    col1, col2, col3 = st.columns([2, 2, 1])

                    with col1:
                        ambient.level_db = st.slider(
                            "Level (dB)",
                            -60.0,
                            -20.0,
                            ambient.level_db,
                            key=f"ambient_level_{scene_key}_{i}",
                        )

                    with col2:
                        if ambient.audio_file and Path(ambient.audio_file).exists():
                            st.text(f"File: {Path(ambient.audio_file).name}")
                            if st.button(
                                "🎵 Play", key=f"play_ambient_{scene_key}_{i}"
                            ):
                                st.audio(ambient.audio_file, format="audio/wav")

                    with col3:
                        if st.button("🗑️ Remove", key=f"remove_ambient_{scene_key}_{i}"):
                            scene.ambient.pop(i)
                            st.rerun()

        return scene

    def _render_audio_playback(self, scene: SceneConfig, scene_key: str):
        """Render audio playback interface for the scene"""
        st.subheader("🔊 Audio Playback & Analysis")

        # Check if scene has been simulated
        cache_key = f"simulated_scene_{scene_key}"
        if cache_key not in st.session_state:
            st.info(
                "Scene not yet simulated. Click 'Simulate Scene' to generate audio."
            )
            return

        mic_signals, metadata = st.session_state[cache_key]

        # Playback controls
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            st.subheader("🎙️ Microphone Array Playback")
            selected_mic = st.selectbox(
                "Select Microphone",
                [0, 1, 2, 3],
                format_func=lambda x: f"Mic {x+1} ({'Left' if x==0 else 'Back' if x==1 else 'Right' if x==2 else 'Front'})",
                key=f"mic_select_{scene_key}",
            )

            if selected_mic is not None:
                st.audio(
                    mic_signals[selected_mic],
                    format="audio/wav",
                    sample_rate=metadata["sample_rate"],
                )

        with col2:
            st.subheader("📊 All Microphones")
            if st.button("Play All Mics", key=f"play_all_{scene_key}"):
                for i in range(4):
                    st.text(f"Microphone {i+1}:")
                    st.audio(
                        mic_signals[i],
                        format="audio/wav",
                        sample_rate=metadata["sample_rate"],
                    )

        with col3:
            st.subheader("💾 Export Options")
            if st.button("Export Audio Files", key=f"export_{scene_key}"):
                self._export_scene_audio(scene, mic_signals, metadata, scene_key)

    def _render_scene_analysis(self, scene: SceneConfig, scene_key: str):
        """Render scene analysis and statistics"""
        st.subheader("📈 Scene Analysis")

        # Basic scene statistics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Duration", f"{scene.duration:.1f}s")
            st.metric("Directional Sounds", len(scene.sounds))
            st.metric("Ambient Layers", len(scene.ambient))

        with col2:
            if scene.sounds:
                distances = [s.distance for s in scene.sounds]
                st.metric("Avg Distance", f"{np.mean(distances):.1f}m")
                st.metric("Min Distance", f"{min(distances):.1f}m")
                st.metric("Max Distance", f"{max(distances):.1f}m")

        with col3:
            if scene.sounds:
                volumes = [s.volume for s in scene.sounds]
                st.metric("Avg Volume", f"{np.mean(volumes):.2f}")
                st.metric("Volume Range", f"{min(volumes):.2f} - {max(volumes):.2f}")

        # Sound timeline visualization
        if scene.sounds:
            self._render_timeline_chart(scene, scene_key)

    def _render_timeline_chart(self, scene: SceneConfig, scene_key: str):
        """Create a timeline chart showing when sounds occur"""
        st.subheader("⏰ Sound Timeline")

        # Create timeline data
        timeline_data = []
        for i, sound in enumerate(scene.sounds):
            # Estimate sound duration (assuming 3-5 seconds for animal sounds)
            estimated_duration = min(5.0, scene.duration - sound.start_time)
            timeline_data.append(
                {
                    "Sound": f"{sound.sound_type} ({sound.distance:.0f}m)",
                    "Start": sound.start_time,
                    "End": sound.start_time + estimated_duration,
                    "Type": sound.sound_type,
                    "Volume": sound.volume,
                }
            )

        if timeline_data:
            df = pd.DataFrame(timeline_data)

            # Create Gantt chart
            fig = px.timeline(
                df,
                x_start="Start",
                x_end="End",
                y="Sound",
                color="Type",
                title="Sound Timeline",
                hover_data=["Volume"],
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    def _show_scene_visualization(self, scene: SceneConfig, scene_key: str):
        """Show 3D visualization of the scene"""
        st.subheader("🌍 3D Scene Visualization")

        if not scene.sounds:
            st.info("No sounds to visualize")
            return

        # Create 3D scatter plot
        x_coords = []
        y_coords = []
        z_coords = []
        labels = []
        colors = []
        sizes = []

        color_map = {
            "elephant": "red",
            "bird": "blue",
            "monkey": "green",
            "lion": "orange",
            "bear": "brown",
            "wolf": "gray",
            "dog": "purple",
            "cat": "pink",
            "chicken": "yellow",
            "horse": "black",
            "cow": "white",
            "sheep": "lightblue",
        }

        for sound in scene.sounds:
            # Convert polar to cartesian coordinates
            azimuth_rad = np.radians(sound.azimuth)
            elevation_rad = np.radians(sound.elevation)

            x = sound.distance * np.cos(elevation_rad) * np.cos(azimuth_rad)
            y = sound.distance * np.cos(elevation_rad) * np.sin(azimuth_rad)
            z = sound.distance * np.sin(elevation_rad)

            x_coords.append(x)
            y_coords.append(y)
            z_coords.append(z)
            labels.append(
                f"{sound.sound_type}<br>Dist: {sound.distance:.0f}m<br>Vol: {sound.volume:.1f}"
            )
            colors.append(color_map.get(sound.sound_type, "gray"))
            sizes.append(sound.volume * 10)  # Scale size by volume

        # Create 3D scatter plot
        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=x_coords,
                    y=y_coords,
                    z=z_coords,
                    mode="markers+text",
                    marker=dict(
                        size=sizes,
                        color=colors,
                        opacity=0.8,
                        line=dict(width=2, color="black"),
                    ),
                    text=labels,
                    textposition="top center",
                    hovertemplate="%{text}<extra></extra>",
                )
            ]
        )

        # Add microphone array at origin
        fig.add_trace(
            go.Scatter3d(
                x=[0, -0.032, 0, 0.032, 0],
                y=[0, 0, -0.032, 0, 0.032],
                z=[0, 0, 0, 0, 0],
                mode="markers",
                marker=dict(size=8, color="black", symbol="square"),
                name="Microphone Array",
                hovertemplate="Microphone Array<extra></extra>",
            )
        )

        fig.update_layout(
            title="3D Acoustic Scene Layout",
            scene=dict(
                xaxis_title="X (meters)",
                yaxis_title="Y (meters)",
                zaxis_title="Z (meters)",
                aspectmode="data",
            ),
            height=600,
        )

        st.plotly_chart(fig, use_container_width=True)

    def _render_audio_file_selector(self, key_suffix: str) -> Tuple[Optional[str], str]:
        """Unified audio file selector"""
        col1, col2 = st.columns([2, 1])

        with col1:
            # Dataset selection
            try:
                available_datasets = list(
                    self.dataset_manager.catalog.get_available_datasets()
                )
                if available_datasets:
                    selected_dataset = st.selectbox(
                        "Dataset",
                        ["Any"] + available_datasets,
                        key=f"dataset_{key_suffix}",
                    )
                else:
                    st.info("No datasets available - using manual file path")
                    selected_dataset = "Any"
            except Exception as e:
                st.warning(f"Dataset loading issue: {e}")
                selected_dataset = "Any"

        with col2:
            # Random selection option
            use_random = st.checkbox("Random selection", key=f"random_{key_suffix}")

        if use_random and selected_dataset != "Any":
            return f"random:{selected_dataset}", f"Random from {selected_dataset}"

        # Manual file selection
        audio_file = st.text_input(
            "Audio File Path (optional)", key=f"file_path_{key_suffix}"
        )
        return audio_file if audio_file else None, (
            Path(audio_file).name if audio_file else ""
        )

    def _create_sound_placement(
        self,
        audio_file: str,
        sound_type: str,
        distance: float,
        azimuth: float,
        elevation: float,
        start_time: float,
        volume: float,
    ) -> SoundPlacement:
        """Create a sound placement from parameters"""
        # Convert polar to cartesian coordinates
        azimuth_rad = np.radians(azimuth)
        elevation_rad = np.radians(elevation)

        x = distance * np.cos(elevation_rad) * np.cos(azimuth_rad)
        y = distance * np.cos(elevation_rad) * np.sin(azimuth_rad)
        z = distance * np.sin(elevation_rad)

        return SoundPlacement(
            audio_file=audio_file,
            position=[x, y, z],
            distance=distance,
            azimuth=azimuth,
            elevation=elevation,
            start_time=start_time,
            volume=volume,
            sound_type=sound_type,
        )

    def _simulate_and_cache_scene(self, scene: SceneConfig, scene_key: str):
        """Simulate scene and cache results"""
        try:
            with st.spinner("Simulating acoustic scene..."):
                mic_signals, metadata = self.simulator.simulate_scene(scene)
                st.session_state[f"simulated_scene_{scene_key}"] = (
                    mic_signals,
                    metadata,
                )
            st.success("Scene simulated successfully!")
        except Exception as e:
            st.error(f"Simulation failed: {e}")

    def _play_isolated_sound(
        self, sound: SoundPlacement, scene: SceneConfig, scene_key: str, sound_idx: int
    ):
        """Play an isolated version of a single sound with acoustic simulation"""
        try:
            # Create a temporary scene with only this sound
            # Use the sound's actual duration for timing
            sound_duration = min(10.0, scene.duration)
            
            # IMPORTANT: Create a fresh copy of the sound with current parameters
            updated_sound = SoundPlacement(
                audio_file=sound.audio_file,
                position=sound.position.copy() if hasattr(sound.position, 'copy') else list(sound.position),
                start_time=sound.start_time,
                volume=sound.volume,  # This should be the updated volume!
                sound_type=sound.sound_type,
                distance=sound.distance,
                azimuth=sound.azimuth,
                elevation=sound.elevation
            )
            
            isolated_scene = SceneConfig(
                name=f"Isolated_{sound.sound_type}",
                duration=sound_duration,
                sounds=[updated_sound],  # Use the fresh copy
                ambient=[],  # No ambient for isolation
                sample_rate=scene.sample_rate,
            )
            
            # Debug: Show what volume we're actually using
            st.write(f"🔊 **DEBUG**: Using volume = {updated_sound.volume:.3f} for simulation")

            with st.spinner("Generating isolated sound with spatial acoustics..."):
                # Capture simulation output
                import io
                import sys
                from contextlib import redirect_stdout
                
                output_capture = io.StringIO()
                with redirect_stdout(output_capture):
                    mic_signals, metadata = self.simulator.simulate_scene(isolated_scene)
                
                simulation_output = output_capture.getvalue()
                if simulation_output:
                    with st.expander("🔍 Simulation Debug Output"):
                        st.code(simulation_output)

                # Show debugging information
                st.success(f"🎯 **Isolated Rendering Info:** Volume={sound.volume:.3f}")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write(f"**Position**: ({sound.position[0]:.1f}, {sound.position[1]:.1f}, {sound.position[2]:.1f})m")
                    st.write(f"**Distance**: {sound.distance:.1f}m")
                
                with col2:
                    st.write(f"**Azimuth**: {sound.azimuth:.1f}°")
                    vol_color = "🔊" if sound.volume >= 1.5 else "🔉" if sound.volume >= 0.7 else "🔈"
                    st.write(f"**Volume**: {vol_color} {sound.volume:.2f} ({'Loud' if sound.volume >= 1.5 else 'Normal' if sound.volume >= 0.7 else 'Quiet'})")
                
                with col3:
                    st.write(f"**Scene Duration**: {sound_duration:.1f}s")
                    st.write(f"**Start Time**: {sound.start_time:.1f}s")

                # Play on different microphones to show spatial effects
                st.write("🎧 **Acoustically Simulated Audio** (with distance/position effects):")
                
                # Microphone selection for isolated sound
                mic_choice = st.radio(
                    "Choose microphone to hear spatial differences:",
                    [0, 1, 2, 3],
                    format_func=lambda x: f"Mic {x+1} ({'Left' if x==0 else 'Back' if x==1 else 'Right' if x==2 else 'Front'})",
                    key=f"isolated_mic_{scene_key}_{sound_idx}",
                    horizontal=True
                )
                
                st.audio(
                    mic_signals[mic_choice],
                    format="audio/wav",
                    sample_rate=metadata["sample_rate"],
                )
                
                # Show signal comparison between mics
                st.write("**🔬 Spatial Effect Analysis:**")
                mic_data = []
                total_energy = 0
                for i in range(4):
                    max_amp = np.max(np.abs(mic_signals[i]))
                    rms = np.sqrt(np.mean(mic_signals[i] ** 2))
                    total_energy += rms
                    mic_data.append({
                        "Mic": f"Mic {i+1}",
                        "Position": ["Left", "Back", "Right", "Front"][i],
                        "Max Amp": f"{max_amp:.4f}",
                        "RMS": f"{rms:.5f}",
                        "Relative dB": f"{20*np.log10(rms + 1e-10):.1f}"
                    })
                
                import pandas as pd
                df_mics = pd.DataFrame(mic_data)
                st.dataframe(df_mics, use_container_width=True)
                
                # Volume debugging
                st.write(f"**🔊 Volume Debug Info:**")
                st.write(f"- **Applied Volume Multiplier**: {updated_sound.volume:.3f}")
                st.write(f"- **Total Signal Energy**: {total_energy:.6f}")
                if updated_sound.volume < 0.5:
                    st.error(f"🔈 **Very Low Volume** ({updated_sound.volume:.3f}) - Sound should be much quieter!")
                elif updated_sound.volume > 1.5:
                    st.success(f"🔊 **High Volume** ({updated_sound.volume:.3f}) - Sound should be much louder!")
                else:
                    st.info(f"🔉 **Normal Volume** ({updated_sound.volume:.3f})")
                    
                st.write(f"- **Expected Effect**: {'Amplified' if updated_sound.volume > 1.0 else 'Attenuated' if updated_sound.volume < 1.0 else 'Unchanged'}")
                
                # Show actual vs expected comparison
                if hasattr(self, '_last_volume_test'):
                    ratio = total_energy / self._last_volume_test['energy'] if self._last_volume_test['energy'] > 0 else 1
                    expected_ratio = sound.volume / self._last_volume_test['volume']
                    st.write(f"- **Actual Energy Ratio**: {ratio:.3f}")
                    st.write(f"- **Expected Energy Ratio**: {expected_ratio:.3f}")
                    if abs(ratio - expected_ratio) < 0.1:
                        st.success("✅ Volume scaling working correctly!")
                    else:
                        st.warning("⚠️ Volume scaling may not be applied as expected")
                        
                # Store for next comparison
                self._last_volume_test = {'volume': sound.volume, 'energy': total_energy}
                
                st.caption(f"💡 **Tip**: Different microphones receive different signal levels based on the sound's position ({sound.azimuth:.0f}° azimuth). Try different mics to hear the spatial effect!")
                
                # Volume effect warning/tip
                if sound.volume < 0.3:
                    st.warning(f"⚠️ Volume is very low ({sound.volume:.2f}). You might need to increase system volume to hear differences.")
                elif sound.volume > 2.0:
                    st.info(f"🔊 High volume setting ({sound.volume:.2f}) - sound will be amplified significantly.")

        except Exception as e:
            st.error(f"Could not generate isolated sound: {e}")
            import traceback
            st.code(traceback.format_exc())

    def _export_scene_audio(
        self,
        scene: SceneConfig,
        mic_signals: np.ndarray,
        metadata: Dict,
        scene_key: str,
    ):
        """Export scene audio files"""
        try:
            # Create temporary directory for export
            export_dir = Path("exports") / f"scene_{scene.name.replace(' ', '_')}"
            export_dir.mkdir(parents=True, exist_ok=True)

            # Save microphone files
            for i in range(4):
                mic_file = export_dir / f"mic_{i+1}.wav"
                sf.write(mic_file, mic_signals[i], metadata["sample_rate"])

            # Save metadata
            metadata_file = export_dir / "metadata.json"
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)

            st.success(f"Audio exported to {export_dir}")

        except Exception as e:
            st.error(f"Export failed: {e}")


class BatchSceneManager:
    """Handles batch scene operations with individual scene editing"""

    def __init__(self, dataset_manager: DatasetManager):
        self.dataset_manager = dataset_manager
        self.scene_manager = SceneManager(dataset_manager)

    def render_batch_editor(self, scenes_jsonl_path: str):
        """Render interface for editing batch-generated scenes"""
        st.header("📋 Batch Scene Editor")

        # Load scenes from JSONL
        scenes = self._load_scenes_from_jsonl(scenes_jsonl_path)
        if not scenes:
            st.error("Could not load scenes from JSONL file")
            return

        st.info(f"Loaded {len(scenes)} scenes from {Path(scenes_jsonl_path).name}")

        # Scene selection
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            selected_scene_idx = st.selectbox(
                "Select Scene to Edit",
                range(len(scenes)),
                format_func=lambda x: f"Scene {x+1}: {scenes[x].name}",
                key="batch_scene_select",
            )

        with col2:
            if st.button("🔄 Re-render All"):
                self._rerender_all_scenes(scenes, scenes_jsonl_path)

        with col3:
            if st.button("💾 Save Changes"):
                self._save_scenes_to_jsonl(scenes, scenes_jsonl_path)

        # Edit selected scene
        if selected_scene_idx is not None:
            st.divider()
            updated_scene = self.scene_manager.render_scene_editor(
                scenes[selected_scene_idx], f"batch_{selected_scene_idx}"
            )
            scenes[selected_scene_idx] = updated_scene

        return scenes

    def _load_scenes_from_jsonl(self, jsonl_path: str) -> List[SceneConfig]:
        """Load scenes from JSONL file"""
        scenes = []
        try:
            with open(jsonl_path, "r") as f:
                for line in f:
                    data = json.loads(line.strip())
                    scene = SceneConfig.from_dict(data)
                    scenes.append(scene)
        except Exception as e:
            st.error(f"Error loading JSONL: {e}")

        return scenes

    def _save_scenes_to_jsonl(self, scenes: List[SceneConfig], jsonl_path: str):
        """Save updated scenes back to JSONL"""
        try:
            with open(jsonl_path, "w") as f:
                for scene in scenes:
                    f.write(json.dumps(scene.to_dict()) + "\n")
            st.success("Scenes saved successfully")
        except Exception as e:
            st.error(f"Error saving scenes: {e}")

    def _rerender_all_scenes(self, scenes: List[SceneConfig], jsonl_path: str):
        """Re-render audio for all scenes in the batch"""
        try:
            output_dir = Path(jsonl_path).parent / "rendered_audio"
            output_dir.mkdir(exist_ok=True)

            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, scene in enumerate(scenes):
                status_text.text(f"Rendering scene {i+1}/{len(scenes)}: {scene.name}")

                # Simulate scene
                mic_signals, metadata = self.scene_manager.simulator.simulate_scene(
                    scene
                )

                # Save audio files
                scene_dir = output_dir / f"scene_{i+1:03d}"
                scene_dir.mkdir(exist_ok=True)

                for mic_idx in range(4):
                    mic_file = scene_dir / f"mic_{mic_idx+1}.wav"
                    sf.write(mic_file, mic_signals[mic_idx], metadata["sample_rate"])

                # Save metadata
                metadata_file = scene_dir / "metadata.json"
                with open(metadata_file, "w") as f:
                    json.dump(metadata, f, indent=2)

                progress_bar.progress((i + 1) / len(scenes))

            status_text.text("Rendering complete!")
            st.success(f"All scenes rendered to {output_dir}")

        except Exception as e:
            st.error(f"Rendering failed: {e}")
