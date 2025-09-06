"""
Modular Scene Creator Component for Wildlife Acoustic Simulation
Separated from main app for better organization
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import os
from pathlib import Path
from typing import Optional

from simple_config import SceneConfig, SoundPlacement, AmbientSound
from simple_simulator import SimpleSimulator, load_config_from_yaml
from dataset_manager import DatasetManager


class SceneCreatorUI:
    """Encapsulates all scene creation UI logic"""
    
    def __init__(self, dataset_manager: DatasetManager):
        self.dataset_manager = dataset_manager
        self.simulator = SimpleSimulator(dataset_manager)
        
        # Initialize session state
        if 'current_scene' not in st.session_state:
            st.session_state.current_scene = SceneConfig(
                name="New Scene",
                duration=10.0,
                sounds=[],
                ambient=[AmbientSound("forest", -45.0)]
            )
    
    def render_sidebar(self):
        """Render the sidebar with scene configuration"""
        with st.sidebar:
            st.header("🎯 Scene Setup")
            
            # Basic scene properties
            st.session_state.current_scene.name = st.text_input(
                "Scene Name", 
                st.session_state.current_scene.name
            )
            st.session_state.current_scene.duration = st.slider(
                "Duration (seconds)", 
                1.0, 60.0, 
                st.session_state.current_scene.duration
            )
            st.session_state.current_scene.description = st.text_area(
                "Description", 
                st.session_state.current_scene.description
            )
            
            # Load preset configurations
            st.subheader("📁 Load Preset")
            config_files = list(Path("configs").glob("*.yaml"))
            config_names = [f.stem for f in config_files]
            
            if config_names:
                selected_config = st.selectbox("Choose preset:", ["Custom"] + config_names)
                
                if st.button("Load Preset") and selected_config != "Custom":
                    try:
                        config_file = f"configs/{selected_config}.yaml"
                        st.session_state.current_scene = load_config_from_yaml(config_file, self.dataset_manager)
                        st.success(f"Loaded {selected_config}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error loading preset: {e}")
            
            # Save current scene
            if st.button("💾 Save Scene"):
                save_name = st.session_state.current_scene.name.replace(' ', '_').lower()
                save_path = f"configs/{save_name}.json"
                try:
                    st.session_state.current_scene.save(save_path)
                    st.success(f"Saved to {save_path}")
                except Exception as e:
                    st.error(f"Error saving: {e}")
    
    def render_sound_adder(self):
        """Render the sound addition interface"""
        st.subheader("Add Sound to Scene")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            position = self._render_position_controls()
        
        with col2:
            audio_file, sound_label = self._render_audio_selection()
        
        # Timing and volume controls
        start_time = st.slider("Start Time (seconds)", 0.0, st.session_state.current_scene.duration, 0.0, 0.1)
        volume = st.slider("Volume", 0.1, 2.0, 1.0, 0.1)
        
        # Add sound button
        if st.button("➕ Add Sound to Scene"):
            if audio_file or (self._has_datasets() and self._get_selected_dataset()):
                new_sound = SoundPlacement(
                    audio_file=audio_file or f"random:{self._get_selected_dataset()}",
                    position=position,
                    start_time=start_time,
                    volume=volume,
                    sound_type=sound_label
                )
                st.session_state.current_scene.sounds.append(new_sound)
                st.success(f"Added {sound_label} at ({position[0]:.1f}, {position[1]:.1f}, {position[2]:.1f})")
                st.rerun()
            else:
                st.error("Please select an audio file or dataset")
    
    def _render_position_controls(self):
        """Render position input controls"""
        st.markdown("#### 📍 Position")
        
        pos_method = st.radio("Position Method:", ["Distance & Angle", "XYZ Coordinates"])
        
        if pos_method == "Distance & Angle":
            distance = st.slider("Distance (meters)", 10.0, 500.0, 100.0, 10.0)
            azimuth = st.slider("Azimuth (degrees)", 0.0, 360.0, 45.0, 15.0)
            elevation = st.slider("Elevation (degrees)", -30.0, 30.0, 0.0, 5.0)
            
            # Convert to XYZ
            azimuth_rad = np.radians(azimuth)
            elevation_rad = np.radians(elevation)
            x = distance * np.sin(azimuth_rad) * np.cos(elevation_rad)
            y = distance * np.cos(azimuth_rad) * np.cos(elevation_rad)
            z = distance * np.sin(elevation_rad)
            position = (x, y, z)
            
            st.caption(f"XYZ: ({x:.1f}, {y:.1f}, {z:.1f})")
        else:
            x = st.number_input("X (meters)", -500.0, 500.0, 100.0)
            y = st.number_input("Y (meters)", -500.0, 500.0, 100.0)
            z = st.number_input("Z (meters)", -100.0, 100.0, 0.0)
            position = (x, y, z)
            
            distance = np.sqrt(x**2 + y**2 + z**2)
            azimuth = np.degrees(np.arctan2(x, y))
            if azimuth < 0:
                azimuth += 360
            st.caption(f"Distance: {distance:.1f}m, Azimuth: {azimuth:.1f}°")
        
        return position
    
    def _render_audio_selection(self):
        """Render audio file selection interface"""
        st.markdown("#### 🎵 Audio Selection")
        
        # Available datasets
        datasets = list(self.dataset_manager.catalog.catalog["datasets"].keys())
        selected_dataset = st.selectbox("Dataset:", datasets if datasets else ["No datasets available"])
        
        if not datasets:
            st.error("No datasets available. Please add datasets first using the dataset manager.")
            return None, "No Dataset"
        
        # Get all files from selected dataset
        all_files = []
        for file_hash, file_info in self.dataset_manager.catalog.catalog["files"].items():
            if file_info.get("dataset") == selected_dataset:
                file_path = file_info["absolute_path"]
                file_name = Path(file_path).name
                duration = file_info.get("duration", 0)
                all_files.append({
                    "name": file_name,
                    "path": file_path,
                    "duration": duration,
                    "info": f"{file_name} ({duration:.1f}s)"
                })
        
        if all_files:
            st.success(f"Found {len(all_files)} files in {selected_dataset}")
            
            # File selection options
            file_selection_method = st.radio("File Selection:", ["Choose Specific File", "Random from Dataset"])
            
            if file_selection_method == "Choose Specific File":
                return self._render_specific_file_selection(all_files)
            else:
                # Random selection
                st.info("🎲 A random file will be selected from this dataset during simulation")
                return None, f"Random from {selected_dataset}"
        else:
            st.warning(f"No audio files found in dataset '{selected_dataset}'")
            return None, "No files available"
    
    def _render_specific_file_selection(self, all_files):
        """Render specific file selection with search"""
        # Show files with duration info
        file_options = [f["info"] for f in all_files[:50]]  # Limit for performance
        if len(all_files) > 50:
            st.info(f"Showing first 50 of {len(all_files)} files. Use search to find specific files.")
        
        # Add search functionality
        search_term = st.text_input("🔍 Search files:", placeholder="Type to filter files...")
        
        if search_term:
            filtered_files = [f for f in all_files if search_term.lower() in f["name"].lower()]
            file_options = [f["info"] for f in filtered_files[:50]]
            st.info(f"Found {len(filtered_files)} matching files")
        else:
            filtered_files = all_files[:50]
        
        if file_options:
            selected_file_info = st.selectbox("Choose Audio File:", file_options)
            # Find the corresponding file path
            selected_file_name = selected_file_info.split(" (")[0]  # Extract name before duration
            audio_file = next((f["path"] for f in (filtered_files if search_term else all_files) 
                             if f["name"] == selected_file_name), None)
            sound_label = selected_file_name
            
            # Add audio preview
            if audio_file and os.path.exists(audio_file):
                st.markdown("#### 🎧 Audio Preview")
                
                # File info first
                file_info = next((f for f in (filtered_files if search_term else all_files) 
                                if f["name"] == selected_file_name), None)
                if file_info:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.caption(f"📏 **Duration:** {file_info['duration']:.1f}s")
                    with col2:
                        st.caption(f"📁 **Size:** {os.path.getsize(audio_file) / 1024:.1f} KB")
                    with col3:
                        # Determine file format for better audio handling
                        file_ext = Path(audio_file).suffix.lower()
                        format_map = {'.wav': 'audio/wav', '.mp3': 'audio/mpeg', '.flac': 'audio/flac'}
                        audio_format = format_map.get(file_ext, 'audio/wav')
                        st.caption(f"🎵 **Format:** {file_ext.upper()}")
                
                try:
                    # Load audio file and play it with appropriate format
                    file_ext = Path(audio_file).suffix.lower()
                    format_map = {'.wav': 'audio/wav', '.mp3': 'audio/mpeg', '.flac': 'audio/flac'}
                    audio_format = format_map.get(file_ext, 'audio/wav')
                    
                    st.audio(audio_file, format=audio_format)
                    st.caption("🎵 Click play button above to preview this sound")
                    
                except Exception as e:
                    st.warning(f"Could not load audio preview: {e}")
                    st.caption("This file might not be compatible with web playback")
            
            return audio_file, sound_label
        else:
            return None, "No matching files"
    
    def _has_datasets(self):
        """Check if any datasets are available"""
        return bool(self.dataset_manager.catalog.catalog["datasets"])
    
    def _get_selected_dataset(self):
        """Get the currently selected dataset"""
        datasets = list(self.dataset_manager.catalog.catalog["datasets"].keys())
        return datasets[0] if datasets else None
    
    def render_scene_overview(self):
        """Render the current scene overview"""
        st.subheader("🎵 Sounds in Scene")
        
        if st.session_state.current_scene.sounds:
            # Create a dataframe for display
            sounds_data = []
            for i, sound in enumerate(st.session_state.current_scene.sounds):
                # Handle display of file info
                if sound.audio_file.startswith("random:"):
                    file_display = f"Random from {sound.audio_file.split(':', 1)[1]}"
                elif sound.audio_file:
                    file_display = Path(sound.audio_file).name
                else:
                    file_display = "Not specified"
                
                sounds_data.append({
                    "Index": i,
                    "Audio Source": sound.sound_type,
                    "Distance": f"{sound.distance:.1f}m",
                    "Azimuth": f"{sound.azimuth:.1f}°",
                    "Start": f"{sound.start_time:.1f}s",
                    "Volume": f"{sound.volume:.1f}",
                    "File": file_display
                })
            
            df = pd.DataFrame(sounds_data)
            st.dataframe(df, use_container_width=True)
            
            # Sound management section
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Preview sound from scene
                preview_idx = st.selectbox("Preview Sound (by Index):", range(len(st.session_state.current_scene.sounds)))
                if st.button("🎧 Preview Selected Sound"):
                    selected_sound = st.session_state.current_scene.sounds[preview_idx]
                    if (selected_sound.audio_file and 
                        not selected_sound.audio_file.startswith("random:") and 
                        os.path.exists(selected_sound.audio_file)):
                        
                        st.markdown("#### 🎧 Sound Preview")
                        st.caption(f"**{selected_sound.sound_type}** at ({selected_sound.position[0]:.1f}m, {selected_sound.position[1]:.1f}m, {selected_sound.position[2]:.1f}m)")
                        st.audio(selected_sound.audio_file, format='audio/wav')
                    else:
                        st.warning("Cannot preview this sound (random selection or file not found)")
            
            with col2:
                # Remove sound
                remove_idx = st.selectbox("Remove Sound (by Index):", range(len(st.session_state.current_scene.sounds)))
                if st.button("🗑️ Remove Selected Sound"):
                    del st.session_state.current_scene.sounds[remove_idx]
                    st.success("Sound removed")
                    st.rerun()
        else:
            st.info("No sounds in scene yet. Add some above!")
    
    def render_3d_visualization(self):
        """Render 3D scene visualization"""
        st.subheader("🗺️ Scene Visualization")
        
        if st.session_state.current_scene.sounds:
            fig = go.Figure()
            
            # Add microphone array at origin
            fig.add_trace(go.Scatter3d(
                x=[0], y=[0], z=[0],
                mode='markers+text',
                marker=dict(size=15, color='red', symbol='diamond'),
                text=['Mic Array'],
                textposition='top center',
                name='Microphone Array'
            ))
            
            # Add sounds
            colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive']
            for i, sound in enumerate(st.session_state.current_scene.sounds):
                color = colors[i % len(colors)]
                fig.add_trace(go.Scatter3d(
                    x=[sound.position[0]],
                    y=[sound.position[1]],
                    z=[sound.position[2]],
                    mode='markers+text',
                    marker=dict(size=10, color=color),
                    text=[f"{sound.sound_type}<br>{sound.distance:.0f}m"],
                    textposition='top center',
                    name=f"{sound.sound_type} {i+1}"
                ))
            
            # Add range circles
            theta = np.linspace(0, 2*np.pi, 100)
            for radius in [50, 100, 200]:
                x_circle = radius * np.cos(theta)
                y_circle = radius * np.sin(theta)
                z_circle = np.zeros_like(x_circle)
                
                fig.add_trace(go.Scatter3d(
                    x=x_circle, y=y_circle, z=z_circle,
                    mode='lines',
                    line=dict(color='lightgray', width=2, dash='dash'),
                    showlegend=False,
                    hoverinfo='skip'
                ))
            
            fig.update_layout(
                title="3D Scene Layout (Mic Array at Origin)",
                scene=dict(
                    xaxis_title="X (meters) - East/West",
                    yaxis_title="Y (meters) - North/South", 
                    zaxis_title="Z (meters) - Up/Down",
                    aspectmode='cube',
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
                ),
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Sounds", len(st.session_state.current_scene.sounds))
            with col2:
                distances = [s.distance for s in st.session_state.current_scene.sounds]
                st.metric("Avg Distance", f"{np.mean(distances):.1f}m")
            with col3:
                st.metric("Scene Duration", f"{st.session_state.current_scene.duration:.1f}s")
        else:
            st.info("Add sounds to see the scene visualization")
    
    def render_simulation_controls(self):
        """Render simulation execution controls"""
        st.subheader("▶️ Run Simulation")
        
        if st.session_state.current_scene.sounds:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("#### 🎛️ Simulation Settings")
                output_dir = st.text_input("Output Directory", "outputs/simulation")
                run_simulation = st.button("🚀 Run Simulation", type="primary")
            
            with col2:
                st.markdown("#### 📊 Scene Summary")
                st.write(f"**Name:** {st.session_state.current_scene.name}")
                st.write(f"**Duration:** {st.session_state.current_scene.duration}s")
                st.write(f"**Sounds:** {len(st.session_state.current_scene.sounds)}")
                st.write(f"**Ambient:** {len(st.session_state.current_scene.ambient)} conditions")
            
            if run_simulation:
                return self._run_simulation(output_dir)
        else:
            st.info("Add some sounds to the scene before running simulation")
        
        return None
    
    def _run_simulation(self, output_dir):
        """Execute the simulation"""
        with st.spinner("Running acoustic simulation..."):
            try:
                # Run simulation
                mic_signals, metadata = self.simulator.simulate_scene(st.session_state.current_scene)
                
                # Save results
                output_path = self.simulator.save_results(mic_signals, metadata, output_dir)
                
                st.success(f"✅ Simulation complete! Results saved to: {output_path}")
                
                # Show results
                self._display_simulation_results(mic_signals, metadata)
                
                return mic_signals, metadata
                
            except Exception as e:
                st.error(f"Simulation failed: {e}")
                import traceback
                st.code(traceback.format_exc())
                return None
    
    def _display_simulation_results(self, mic_signals, metadata):
        """Display simulation results"""
        st.subheader("📈 Results Preview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Output Shape", f"{mic_signals.shape[0]} mics × {mic_signals.shape[1]} samples")
            st.metric("Max Amplitude", f"{np.max(np.abs(mic_signals)):.3f}")
            st.metric("Duration", f"{mic_signals.shape[1] / metadata['sample_rate']:.1f}s")
        
        with col2:
            # Microphone selection for audio playback
            st.write("**🎧 Microphone Audio Playback:**")
            selected_mic = st.selectbox("Select Microphone:", 
                                      options=[0, 1, 2, 3], 
                                      format_func=lambda x: f"Microphone {x+1} ({'Left' if x==0 else 'Back' if x==1 else 'Right' if x==2 else 'Front'})")
            st.audio(mic_signals[selected_mic], format='audio/wav', sample_rate=metadata['sample_rate'])
        
        # Show all microphone audio players
        st.subheader("🎧 All Microphone Channels")
        st.caption("ReSpeaker USB 4-Mic Array - Compare audio from different microphone positions")
        
        mic_names = ["Left", "Back", "Right", "Front"]
        mic_colors = ["🔴", "🟡", "🟢", "🔵"]  # Visual indicators
        
        cols = st.columns(4)
        for i, (col, name, color) in enumerate(zip(cols, mic_names, mic_colors)):
            with col:
                st.markdown(f"**{color} Mic {i+1} ({name})**")
                st.audio(mic_signals[i], format='audio/wav', sample_rate=metadata['sample_rate'])
        
        # Show waveform comparison
        st.subheader("📊 Waveform Analysis")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        
        time_axis = np.arange(mic_signals.shape[1]) / metadata['sample_rate']
        
        for i in range(4):
            axes[i].plot(time_axis, mic_signals[i])
            axes[i].set_title(f"{mic_colors[i]} Microphone {i+1} ({mic_names[i]})")
            axes[i].set_xlabel("Time (s)")
            axes[i].set_ylabel("Amplitude")
            axes[i].grid(True, alpha=0.3)
            
            # Add RMS level info
            rms_level = np.sqrt(np.mean(mic_signals[i]**2))
            axes[i].text(0.02, 0.98, f"RMS: {rms_level:.4f}", 
                        transform=axes[i].transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Show microphone array analysis
        st.subheader("🔬 Microphone Array Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📊 Signal Levels (RMS)**")
            rms_data = []
            for i in range(4):
                rms = np.sqrt(np.mean(mic_signals[i]**2))
                rms_data.append({
                    "Microphone": f"{mic_colors[i]} Mic {i+1} ({mic_names[i]})",
                    "RMS Level": f"{rms:.6f}",
                    "dB": f"{20 * np.log10(rms + 1e-10):.1f}"
                })
            
            df_rms = pd.DataFrame(rms_data)
            st.dataframe(df_rms, use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("**🎯 Array Geometry**")
            st.caption("ReSpeaker USB 4-Mic Array positions relative to center:")
            geometry_data = [
                {"Mic": "🔴 Mic 1 (Left)", "Position": "(-32, 0, 0) mm"},
                {"Mic": "🟡 Mic 2 (Back)", "Position": "(0, -32, 0) mm"}, 
                {"Mic": "🟢 Mic 3 (Right)", "Position": "(32, 0, 0) mm"},
                {"Mic": "🔵 Mic 4 (Front)", "Position": "(0, 32, 0) mm"}
            ]
            df_geom = pd.DataFrame(geometry_data)
            st.dataframe(df_geom, use_container_width=True, hide_index=True)
        
        st.info("💡 **Tip**: Different microphones will receive slightly different signals based on sound source positions. Listen to each microphone to hear how sound directivity affects the recording!")
        


def render_scene_creator_tab(dataset_manager: DatasetManager):
    """Main function to render the scene creator tab"""
    scene_creator = SceneCreatorUI(dataset_manager)
    
    # Render sidebar
    scene_creator.render_sidebar()
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["🎵 Add Sounds", "🗺️ Scene View", "▶️ Simulate"])
    
    with tab1:
        scene_creator.render_sound_adder()
        scene_creator.render_scene_overview()
    
    with tab2:
        scene_creator.render_3d_visualization()
    
    with tab3:
        scene_creator.render_simulation_controls()