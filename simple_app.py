"""
Simplified Streamlit Interface for Wildlife Acoustic Scene Creation
Focus on placing sounds in 3D space and running simulations
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from pathlib import Path
import json
import time

# Import simplified modules
from simple_config import SceneConfig, SoundPlacement, AmbientSound, SceneBuilder
from simple_simulator import SimpleSimulator, load_config_from_yaml
from dataset_manager import DatasetManager

# Page config
st.set_page_config(
    page_title="Wildlife Acoustic Scene Creator",
    page_icon="🎙️",
    layout="wide"
)

st.title("🎙️ Wildlife Acoustic Scene Creator")
st.markdown("Create realistic acoustic scenes using your audio datasets")

# Initialize dataset manager
@st.cache_resource
def init_dataset_manager():
    return DatasetManager()

dataset_manager = init_dataset_manager()

# Initialize session state
if 'current_scene' not in st.session_state:
    st.session_state.current_scene = SceneConfig(
        name="New Scene",
        duration=10.0,
        sounds=[],
        ambient=[AmbientSound("forest", -45.0)]
    )

# Sidebar for scene configuration
with st.sidebar:
    st.header("🎯 Scene Setup")
    
    # Basic scene properties
    st.session_state.current_scene.name = st.text_input("Scene Name", st.session_state.current_scene.name)
    st.session_state.current_scene.duration = st.slider("Duration (seconds)", 1.0, 60.0, st.session_state.current_scene.duration)
    st.session_state.current_scene.description = st.text_area("Description", st.session_state.current_scene.description)
    
    # Load preset configurations
    st.subheader("📁 Load Preset")
    config_files = list(Path("configs").glob("*.yaml"))
    config_names = [f.stem for f in config_files]
    
    if config_names:
        selected_config = st.selectbox("Choose preset:", ["Custom"] + config_names)
        
        if st.button("Load Preset") and selected_config != "Custom":
            try:
                config_file = f"configs/{selected_config}.yaml"
                st.session_state.current_scene = load_config_from_yaml(config_file, dataset_manager)
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

# Main content area
tab1, tab2, tab3 = st.tabs(["🎵 Add Sounds", "🗺️ Scene View", "▶️ Simulate"])

with tab1:
    st.subheader("Add Sound to Scene")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### 📍 Position")
        
        # Position input methods
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
    
    with col2:
        st.markdown("#### 🎵 Audio Selection")
        
        # Available datasets
        datasets = list(dataset_manager.catalog.catalog["datasets"].keys())
        selected_dataset = st.selectbox("Dataset:", datasets if datasets else ["No datasets available"])
        
        if not datasets:
            st.error("No datasets available. Please add datasets first using the dataset manager.")
            audio_file = None
            sound_label = "No Dataset"
        else:
            # Get all files from selected dataset
            all_files = []
            for file_hash, file_info in dataset_manager.catalog.catalog["files"].items():
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
                    # Show files with duration info
                    file_options = [f["info"] for f in all_files[:50]]  # Limit to first 50 for performance
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
                    else:
                        audio_file = None
                        sound_label = "No matching files"
                else:
                    # Random selection
                    audio_file = None  # Will be selected randomly during simulation
                    sound_label = f"Random from {selected_dataset}"
                    st.info("🎲 A random file will be selected from this dataset during simulation")
            else:
                st.warning(f"No audio files found in dataset '{selected_dataset}'")
                audio_file = None
                sound_label = "No files available"
        
        # Timing and volume
        start_time = st.slider("Start Time (seconds)", 0.0, st.session_state.current_scene.duration, 0.0, 0.1)
        volume = st.slider("Volume", 0.1, 2.0, 1.0, 0.1)
        
        # Add sound button
        if st.button("➕ Add Sound to Scene"):
            if audio_file or (datasets and selected_dataset):
                new_sound = SoundPlacement(
                    audio_file=audio_file or f"random:{selected_dataset}",  # Use special syntax for random selection
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
    
    # Current sounds in scene
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
        st.dataframe(df, width='stretch')
        
        # Remove sound
        remove_idx = st.selectbox("Remove Sound (by Index):", range(len(st.session_state.current_scene.sounds)))
        if st.button("🗑️ Remove Selected Sound"):
            del st.session_state.current_scene.sounds[remove_idx]
            st.success("Sound removed")
            st.rerun()
    else:
        st.info("No sounds in scene yet. Add some above!")

with tab2:
    st.subheader("🗺️ Scene Visualization")
    
    if st.session_state.current_scene.sounds:
        # Create 3D plot
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
        
        st.plotly_chart(fig, width='stretch')
        
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

with tab3:
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
            # Run the simulation
            with st.spinner("Running acoustic simulation..."):
                try:
                    # Create simulator
                    simulator = SimpleSimulator(dataset_manager)
                    
                    # Run simulation
                    mic_signals, metadata = simulator.simulate_scene(st.session_state.current_scene)
                    
                    # Save results
                    output_path = simulator.save_results(mic_signals, metadata, output_dir)
                    
                    st.success(f"✅ Simulation complete! Results saved to: {output_path}")
                    
                    # Show some results
                    st.subheader("📈 Results Preview")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Output Shape", f"{mic_signals.shape[0]} mics × {mic_signals.shape[1]} samples")
                        st.metric("Max Amplitude", f"{np.max(np.abs(mic_signals)):.3f}")
                        st.metric("Duration", f"{mic_signals.shape[1] / metadata['sample_rate']:.1f}s")
                    
                    with col2:
                        # Play audio from first microphone
                        st.write("**🎧 Mic 1 Audio:**")
                        st.audio(mic_signals[0], format='audio/wav', sample_rate=metadata['sample_rate'])
                    
                    # Show waveform
                    import matplotlib.pyplot as plt
                    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
                    axes = axes.flatten()
                    
                    time_axis = np.arange(mic_signals.shape[1]) / metadata['sample_rate']
                    
                    for i in range(4):
                        axes[i].plot(time_axis, mic_signals[i])
                        axes[i].set_title(f"Microphone {i+1}")
                        axes[i].set_xlabel("Time (s)")
                        axes[i].set_ylabel("Amplitude")
                        axes[i].grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"Simulation failed: {e}")
                    import traceback
                    st.code(traceback.format_exc())
    else:
        st.info("Add some sounds to the scene before running simulation")

# Footer
st.markdown("---")
st.markdown("🎙️ **Wildlife Acoustic Scene Creator** - Create realistic microphone array data for algorithm validation")