"""
Streamlit UI for the Scene Generator
"""

import streamlit as st
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, Dict, List
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from scene_generator import SceneGenerator, GeneratorConfig, SceneRenderer
from dataset_manager import DatasetManager
from scene_manager import BatchSceneManager

class SceneGeneratorUI:
    """Streamlit UI for batch scene generation"""
    
    def __init__(self, dataset_manager: DatasetManager):
        self.dataset_manager = dataset_manager
        self.batch_scene_manager = BatchSceneManager(dataset_manager)
    
    def render(self):
        """Render the scene generator interface"""
        st.header("🤖 Automated Scene Generator")
        st.markdown("""
        Generate large datasets of elephant detection scenes with realistic acoustic modeling.
        Perfect for training machine learning models for wildlife monitoring.
        """)
        
        # Create tabs for different functions
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Generate Scenes", "🎵 Render Audio", "🎬 Edit Scenes", "📈 View Statistics", "ℹ️ Help"])
        
        with tab1:
            self._render_generation_tab()
        
        with tab2:
            self._render_rendering_tab()
        
        with tab3:
            self._render_batch_editor_tab()
        
        with tab4:
            self._render_statistics_tab()
        
        with tab5:
            self._render_help_tab()
    
    def _render_generation_tab(self):
        """Render the scene generation tab"""
        st.subheader("🎬 Generate Scene Dataset")
        
        # Configuration parameters
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### 📊 Dataset Configuration")
            
            # Dataset naming
            dataset_name = st.text_input(
                "Dataset Name",
                value="elephant_detection_dataset",
                help="Name for this dataset - will be used for file naming"
            )
            
            num_samples = st.number_input(
                "Number of Scenes",
                min_value=1,
                max_value=10000,
                value=100,
                step=1,
                help="Total number of acoustic scenes to generate"
            )
            
            positive_ratio = st.slider(
                "Positive Sample Ratio",
                min_value=0.0,
                max_value=1.0,
                value=0.4,
                step=0.05,
                help="Percentage of scenes that contain elephants (0.4 = 40%)"
            )
            
            scene_duration = st.number_input(
                "Scene Duration (seconds)",
                min_value=5.0,
                max_value=300.0,
                value=30.0,
                step=5.0,
                help="Duration of each generated scene"
            )
        
        with col2:
            st.markdown("#### 🎵 Sound Configuration")
            
            col2a, col2b = st.columns([1, 1])
            with col2a:
                min_directional = st.number_input("Min Directional Sounds", 1, 100, 10, key="min_dir")
                min_ambient = st.number_input("Min Ambient Sounds", 1, 20, 3, key="min_amb")
                min_distance = st.number_input("Min Distance (m)", 1.0, 1000.0, 10.0, key="min_dist")
            
            with col2b:
                max_directional = st.number_input("Max Directional Sounds", 1, 100, 50, key="max_dir")
                max_ambient = st.number_input("Max Ambient Sounds", 1, 20, 5, key="max_amb")
                max_distance = st.number_input("Max Distance (m)", 1.0, 1000.0, 500.0, key="max_dist")
        
        # Sound type selection
        st.markdown("#### 🎵 Sound Type Selection")
        
        col3a, col3b = st.columns([1, 1])
        
        with col3a:
            st.markdown("**Directional Sound Types:**")
            
            # Get available directional sound types from the generator
            directional_types = sorted(list(SceneGenerator.DIRECTIONAL_TYPES))
            
            # Add "Select All" / "Deselect All" buttons
            col_all, col_none = st.columns([1, 1])
            with col_all:
                select_all_dir = st.button("✅ Select All", key="select_all_directional")
            with col_none:
                deselect_all_dir = st.button("❌ Deselect All", key="deselect_all_directional")
            
            # Initialize session state for directional types selection
            if 'selected_directional_types' not in st.session_state:
                st.session_state.selected_directional_types = directional_types.copy()
            
            # Handle select/deselect all buttons
            if select_all_dir:
                st.session_state.selected_directional_types = directional_types.copy()
                st.rerun()
            if deselect_all_dir:
                st.session_state.selected_directional_types = []
                st.rerun()
            
            # Scrollable container for directional sound types
            with st.container(height=300):
                selected_directional_types = []
                for sound_type in directional_types:
                    is_selected = st.checkbox(
                        f"🎯 {sound_type.title()}",
                        value=sound_type in st.session_state.selected_directional_types,
                        key=f"dir_{sound_type}"
                    )
                    if is_selected:
                        selected_directional_types.append(sound_type)
                
                # Update session state
                st.session_state.selected_directional_types = selected_directional_types
        
        with col3b:
            st.markdown("**Ambient Sound Types:**")
            
            # Get available ambient sound types from the generator
            ambient_types = sorted(list(SceneGenerator.AMBIENT_TYPES))
            
            # Add "Select All" / "Deselect All" buttons
            col_all_amb, col_none_amb = st.columns([1, 1])
            with col_all_amb:
                select_all_amb = st.button("✅ Select All", key="select_all_ambient")
            with col_none_amb:
                deselect_all_amb = st.button("❌ Deselect All", key="deselect_all_ambient")
            
            # Initialize session state for ambient types selection
            if 'selected_ambient_types' not in st.session_state:
                st.session_state.selected_ambient_types = ambient_types.copy()
            
            # Handle select/deselect all buttons
            if select_all_amb:
                st.session_state.selected_ambient_types = ambient_types.copy()
                st.rerun()
            if deselect_all_amb:
                st.session_state.selected_ambient_types = []
                st.rerun()
            
            # Scrollable container for ambient sound types
            with st.container(height=300):
                selected_ambient_types = []
                for sound_type in ambient_types:
                    is_selected = st.checkbox(
                        f"🌿 {sound_type.title()}",
                        value=sound_type in st.session_state.selected_ambient_types,
                        key=f"amb_{sound_type}"
                    )
                    if is_selected:
                        selected_ambient_types.append(sound_type)
                
                # Update session state
                st.session_state.selected_ambient_types = selected_ambient_types
        
        # Show selection summary
        st.markdown("#### 📊 Selection Summary")
        col_summary1, col_summary2 = st.columns([1, 1])
        with col_summary1:
            st.info(f"**Selected Directional Types:** {len(st.session_state.selected_directional_types)} of {len(directional_types)}")
        with col_summary2:
            st.info(f"**Selected Ambient Types:** {len(st.session_state.selected_ambient_types)} of {len(ambient_types)}")
        
        # Validation
        if min_directional > max_directional:
            st.error("Minimum directional sounds cannot be greater than maximum")
            return
        
        if min_ambient > max_ambient:
            st.error("Minimum ambient sounds cannot be greater than maximum")
            return
        
        if min_distance > max_distance:
            st.error("Minimum distance cannot be greater than maximum distance")
            return
        
        if len(st.session_state.selected_directional_types) == 0:
            st.error("Please select at least one directional sound type")
            return
        
        if len(st.session_state.selected_ambient_types) == 0:
            st.warning("No ambient sound types selected - scenes will have no background ambience")
        
        # Output configuration
        output_dir = st.text_input(
            "Output Directory",
            value="outputs/generated_scenes",
            help="Directory where scene configurations will be saved"
        )
        
        # Display statistics preview
        num_positive = int(num_samples * positive_ratio)
        num_negative = num_samples - num_positive
        
        st.markdown("#### 📊 Generation Preview")
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        
        with col1:
            st.metric("Total Scenes", num_samples)
        with col2:
            st.metric("Positive Scenes", f"{num_positive} ({positive_ratio:.1%})")
        with col3:
            st.metric("Negative Scenes", f"{num_negative} ({1-positive_ratio:.1%})")
        with col4:
            st.metric("Duration", f"{scene_duration}s each")
        
        # Generation button
        if st.button("🚀 Generate Dataset", type="primary"):
            self._generate_scenes(
                dataset_name, num_samples, positive_ratio, scene_duration,
                min_directional, max_directional, min_ambient, max_ambient,
                min_distance, max_distance, output_dir,
                st.session_state.selected_directional_types,
                st.session_state.selected_ambient_types
            )
    
    def _generate_scenes(self, dataset_name, num_samples, positive_ratio, duration,
                        min_dir, max_dir, min_amb, max_amb,
                        min_dist, max_dist, output_dir,
                        selected_directional_types, selected_ambient_types):
        """Generate scenes with progress tracking"""
        
        with st.spinner("Initializing scene generator..."):
            config = GeneratorConfig(
                dataset_name=dataset_name,
                num_samples=num_samples,
                positive_ratio=positive_ratio,
                scene_duration=duration,
                min_directional_sounds=min_dir,
                max_directional_sounds=max_dir,
                min_ambient_sounds=min_amb,
                max_ambient_sounds=max_amb,
                min_distance=min_dist,
                max_distance=max_dist,
                output_dir=output_dir,
                allowed_directional_types=selected_directional_types,
                allowed_ambient_types=selected_ambient_types
            )
            
            generator = SceneGenerator(self.dataset_manager, config)
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("🎬 Generating scenes...")
            
            # Generate the dataset
            jsonl_path = generator.generate_dataset()
            
            progress_bar.progress(100)
            status_text.text("✅ Generation completed!")
            
            # Show results
            st.success(f"🎉 Successfully generated {num_samples} scenes!")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("#### 📁 Generated Files")
                st.code(f"JSONL: {jsonl_path}")
                summary_path = Path(jsonl_path).parent / f"summary_{num_samples}.json"
                st.code(f"Summary: {summary_path}")
            
            with col2:
                st.markdown("#### 🎵 Next Steps")
                st.info("Use the 'Render Audio' tab to generate 4-microphone audio files from these configurations.")
            
            # Store generated path in session state
            st.session_state.last_generated_jsonl = jsonl_path
            
        except Exception as e:
            st.error(f"❌ Generation failed: {str(e)}")
            progress_bar.empty()
            status_text.empty()
    
    def _render_rendering_tab(self):
        """Render the audio rendering tab"""
        st.subheader("🎵 Render 4-Microphone Audio")
        
        st.markdown("""
        Convert scene configurations (JSONL files) into realistic 4-channel microphone array audio.
        Each scene generates 4 WAV files corresponding to the ReSpeaker microphone positions.
        """)
        
        # File selection
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Check for recently generated files
            recent_files = []
            if hasattr(st.session_state, 'last_generated_jsonl'):
                recent_files = [st.session_state.last_generated_jsonl]
            
            # Find existing JSONL files
            outputs_dir = Path("outputs/generated_scenes")
            if outputs_dir.exists():
                existing_files = []
                # Look for JSONL files in dataset subdirectories
                for dataset_dir in outputs_dir.iterdir():
                    if dataset_dir.is_dir():
                        existing_files.extend(dataset_dir.glob("*.jsonl"))
                recent_files.extend([str(f) for f in existing_files if str(f) not in recent_files])
            
            if recent_files:
                jsonl_path = st.selectbox(
                    "Select JSONL File",
                    options=recent_files,
                    help="Choose a scene configuration file to render"
                )
            else:
                jsonl_path = st.text_input(
                    "JSONL File Path",
                    placeholder="path/to/generated_scenes.jsonl",
                    help="Path to the JSONL scene configuration file"
                )
        
        with col2:
            if st.button("🔄 Refresh Files"):
                st.rerun()
        
        # Output directory
        if jsonl_path:
            default_output = str(Path(jsonl_path).parent / "rendered_audio")
        else:
            default_output = "rendered_audio"
        
        render_output_dir = st.text_input(
            "Audio Output Directory",
            value=default_output,
            help="Directory where 4-channel audio files will be saved"
        )
        
        # File preview
        if jsonl_path and Path(jsonl_path).exists():
            self._show_jsonl_preview(jsonl_path)
        
        # Render button
        if st.button("🎵 Render Audio", type="primary") and jsonl_path:
            if not Path(jsonl_path).exists():
                st.error("JSONL file not found!")
                return
            
            self._render_audio_files(jsonl_path, render_output_dir)
    
    def _show_jsonl_preview(self, jsonl_path: str):
        """Show preview of JSONL file contents"""
        try:
            # Load all scenes for accurate statistics
            with open(jsonl_path, 'r') as f:
                all_scenes = [json.loads(line) for line in f.readlines()]
            
            st.markdown("#### 📋 File Preview")
            
            # Accurate summary stats from all scenes
            total_scenes = len(all_scenes)
            positive_scenes = sum(1 for s in all_scenes if s.get('metadata', {}).get('is_positive', False))
            negative_scenes = total_scenes - positive_scenes
            
            col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
            
            with col1:
                st.metric("Total Scenes", total_scenes)
            with col2:
                st.metric("Positive Scenes", positive_scenes)
            with col3:
                st.metric("Negative Scenes", negative_scenes)
            with col4:
                positive_ratio = positive_scenes / total_scenes if total_scenes > 0 else 0
                st.metric("Positive Ratio", f"{positive_ratio:.1%}")
            
            if all_scenes:
                # Add tabbed view for preview options
                preview_tab1, preview_tab2 = st.tabs(["📊 Scene Table", "🔍 Sample Scene Details"])
                
                with preview_tab1:
                    self._show_scenes_table(all_scenes)
                
                with preview_tab2:
                    self._show_sample_scene_details(all_scenes[0])
        
        except Exception as e:
            st.warning(f"Could not preview file: {str(e)}")
    
    def _show_scenes_table(self, all_scenes: list):
        """Show scenes data in a table format"""
        # Create DataFrame from scenes
        table_data = []
        
        for i, scene in enumerate(all_scenes):
            metadata = scene.get('metadata', {})
            sounds = scene.get('sounds', [])
            ambient = scene.get('ambient', [])
            
            # Count sound types
            sound_types = {}
            elephant_count = 0
            
            for sound in sounds:
                sound_type = sound.get('sound_type', 'unknown')
                sound_types[sound_type] = sound_types.get(sound_type, 0) + 1
                if sound_type == 'elephant':
                    elephant_count += 1
            
            # Get most common sound types (top 3)
            top_sounds = sorted(sound_types.items(), key=lambda x: x[1], reverse=True)[:3]
            top_sounds_str = ", ".join([f"{stype}({count})" for stype, count in top_sounds])
            
            table_data.append({
                'Scene ID': i + 1,
                'Name': scene.get('name', 'Unknown')[:30] + "..." if len(scene.get('name', '')) > 30 else scene.get('name', 'Unknown'),
                'Type': '✅ Positive' if metadata.get('is_positive', False) else '❌ Negative',
                'Duration (s)': scene.get('duration', 0),
                'Directional Sounds': len(sounds),
                'Ambient Sounds': len(ambient),
                'Elephant Count': elephant_count,
                'Top Sound Types': top_sounds_str if top_sounds_str else 'None'
            })
        
        # Convert to DataFrame and display
        df = pd.DataFrame(table_data)
        
        # Display with filtering options
        st.markdown("##### 🔍 Filter Options")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            show_type = st.selectbox(
                "Show Type",
                options=['All', 'Positive Only', 'Negative Only'],
                key="scene_table_filter_type"
            )
        
        with col2:
            min_sounds = st.number_input(
                "Min Directional Sounds",
                min_value=0,
                max_value=100,
                value=0,
                key="scene_table_min_sounds"
            )
        
        with col3:
            show_elephants_only = st.checkbox(
                "Show scenes with elephants only",
                key="scene_table_elephants_only"
            )
        
        # Apply filters
        filtered_df = df.copy()
        
        if show_type == 'Positive Only':
            filtered_df = filtered_df[filtered_df['Type'] == '✅ Positive']
        elif show_type == 'Negative Only':
            filtered_df = filtered_df[filtered_df['Type'] == '❌ Negative']
        
        if min_sounds > 0:
            filtered_df = filtered_df[filtered_df['Directional Sounds'] >= min_sounds]
        
        if show_elephants_only:
            filtered_df = filtered_df[filtered_df['Elephant Count'] > 0]
        
        st.markdown(f"##### 📊 Scene Data ({len(filtered_df)} of {len(df)} scenes)")
        
        # Display the filtered table
        st.dataframe(
            filtered_df,
            use_container_width=True,
            height=400,
            hide_index=True
        )
        
        # Summary statistics
        if len(filtered_df) > 0:
            col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
            
            with col1:
                positive_count = len(filtered_df[filtered_df['Type'] == '✅ Positive'])
                st.metric("Positive Scenes", positive_count)
            
            with col2:
                avg_directional = filtered_df['Directional Sounds'].mean()
                st.metric("Avg Directional Sounds", f"{avg_directional:.1f}")
            
            with col3:
                total_elephants = filtered_df['Elephant Count'].sum()
                st.metric("Total Elephants", int(total_elephants))
            
            with col4:
                avg_duration = filtered_df['Duration (s)'].mean()
                st.metric("Avg Duration", f"{avg_duration:.1f}s")
    
    def _show_sample_scene_details(self, sample_scene: dict):
        """Show detailed information about a sample scene"""
        st.markdown("##### 📋 Scene Overview")
        
        metadata = sample_scene.get('metadata', {})
        sounds = sample_scene.get('sounds', [])
        ambient = sample_scene.get('ambient', [])
        
        # Basic scene info
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("**Scene Information:**")
            scene_info = {
                "Name": sample_scene.get("name", "Unknown"),
                "Duration": f"{sample_scene.get('duration', 0)}s",
                "Sample Rate": f"{sample_scene.get('sample_rate', 16000)} Hz",
                "Scene Type": "✅ Positive" if metadata.get("contains_elephant") else "❌ Negative",
                "Scene ID": metadata.get("scene_id", "Unknown")
            }
            
            for key, value in scene_info.items():
                st.text(f"• {key}: {value}")
        
        with col2:
            st.markdown("**Sound Counts:**")
            sound_counts = {
                "Directional Sounds": len(sounds),
                "Ambient Sounds": len(ambient),
                "Elephant Sounds": metadata.get("elephant_count", 0),
                "Total Audio Sources": len(sounds) + len(ambient)
            }
            
            for key, value in sound_counts.items():
                st.text(f"• {key}: {value}")
        
        # Directional sounds breakdown
        if sounds:
            st.markdown("##### 🎯 Directional Sounds")
            
            sound_data = []
            for i, sound in enumerate(sounds):
                sound_data.append({
                    '#': i + 1,
                    'Type': sound.get('sound_type', 'unknown'),
                    'Distance (m)': f"{sound.get('distance', 0):.1f}",
                    'Azimuth (°)': f"{sound.get('azimuth', 0):.1f}",
                    'Height (m)': f"{sound.get('position', [0,0,0])[2]:.1f}",
                    'Start Time (s)': f"{sound.get('start_time', 0):.1f}",
                    'Volume': f"{sound.get('volume', 1.0):.2f}"
                })
            
            sounds_df = pd.DataFrame(sound_data)
            st.dataframe(sounds_df, use_container_width=True, hide_index=True)
        
        # Ambient sounds breakdown  
        if ambient:
            st.markdown("##### 🌿 Ambient Sounds")
            
            ambient_data = []
            for i, amb in enumerate(ambient):
                file_name = "Procedural" if not amb.get('audio_file') else Path(amb.get('audio_file', '')).name
                ambient_data.append({
                    '#': i + 1,
                    'Type': amb.get('sound_type', 'unknown'),
                    'Level (dB)': f"{amb.get('level_db', 0):.1f}",
                    'Source': file_name
                })
            
            ambient_df = pd.DataFrame(ambient_data)
            st.dataframe(ambient_df, use_container_width=True, hide_index=True)
    
    def _render_audio_files(self, jsonl_path: str, output_dir: str):
        """Render audio files with progress tracking"""
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Count scenes
            with open(jsonl_path, 'r') as f:
                total_scenes = sum(1 for _ in f)
            
            status_text.text(f"🎵 Rendering {total_scenes} scenes...")
            
            # Initialize renderer
            renderer = SceneRenderer(self.dataset_manager)
            
            # Render with progress updates
            renderer.render_scenes_from_jsonl(jsonl_path, output_dir)
            
            progress_bar.progress(100)
            status_text.text("✅ Rendering completed!")
            
            # Show results
            st.success(f"🎉 Successfully rendered {total_scenes} scenes!")
            
            output_path = Path(output_dir)
            
            st.markdown("#### 📁 Generated Audio Files")
            st.info(f"**Output Directory**: {output_path}")
            st.info(f"**Files per Scene**: 4 WAV files (mic_1.wav, mic_2.wav, mic_3.wav, mic_4.wav) + metadata.json")
            
            # Show directory structure
            if output_path.exists():
                scene_dirs = [d for d in output_path.iterdir() if d.is_dir()]
                st.markdown(f"**Generated {len(scene_dirs)} scene directories**")
                
                # Show first few directories
                for scene_dir in scene_dirs[:3]:
                    st.text(f"📁 {scene_dir.name}")
                    wav_files = list(scene_dir.glob("*.wav"))
                    for wav_file in wav_files:
                        st.text(f"   🎵 {wav_file.name}")
                
                if len(scene_dirs) > 3:
                    st.text(f"   ... and {len(scene_dirs) - 3} more scene directories")
            
        except Exception as e:
            st.error(f"❌ Rendering failed: {str(e)}")
            progress_bar.empty()
            status_text.empty()
    
    def _render_batch_editor_tab(self):
        """Render the batch scene editor tab"""
        st.subheader("🎬 Scene Editor & Playback")
        st.markdown("""
        Select any generated scene file to edit individual scenes, preview audio,
        and make targeted modifications without regenerating the entire batch.
        """)
        
        # File selection
        available_files = []
        outputs_dir = Path("outputs/generated_scenes")
        if outputs_dir.exists():
            # Look for JSONL files in dataset subdirectories
            for dataset_dir in outputs_dir.iterdir():
                if dataset_dir.is_dir():
                    available_files.extend(dataset_dir.glob("*.jsonl"))
        
        if not available_files:
            st.info("No scene files found. Generate scenes first in the 'Generate Scenes' tab.")
            return
        
        selected_file = st.selectbox(
            "Select Scene File to Edit",
            available_files,
            format_func=lambda x: x.name,
            key="batch_editor_file_select"
        )
        
        if selected_file:
            st.info(f"Selected: {selected_file.name}")
            
            # Use BatchSceneManager for editing
            try:
                self.batch_scene_manager.render_batch_editor(str(selected_file))
            except Exception as e:
                st.error(f"Error loading scene editor: {e}")
                st.write("Please ensure the selected file contains valid scene data.")
    
    def _render_statistics_tab(self):
        """Render statistics visualization tab"""
        st.subheader("📈 Dataset Statistics")
        
        # File selection
        summary_files = []
        outputs_dir = Path("outputs/generated_scenes")
        if outputs_dir.exists():
            # Look for summary files in dataset subdirectories
            for dataset_dir in outputs_dir.iterdir():
                if dataset_dir.is_dir():
                    summary_files.extend(dataset_dir.glob("summary_*.json"))
        
        if not summary_files:
            st.info("No summary statistics found. Generate a dataset first to see statistics.")
            return
        
        selected_summary = st.selectbox(
            "Select Summary File",
            options=[str(f) for f in summary_files],
            format_func=lambda x: Path(x).name
        )
        
        if selected_summary:
            self._show_statistics(selected_summary)
    
    def _show_statistics(self, summary_path: str):
        """Display statistics from summary file"""
        try:
            with open(summary_path, 'r') as f:
                stats = json.load(f)
            
            # Overview metrics
            st.markdown("#### 📊 Dataset Overview")
            col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
            
            with col1:
                st.metric("Total Scenes", stats["total_scenes"])
            with col2:
                st.metric("Positive Scenes", stats["positive_scenes"])
            with col3:
                st.metric("Negative Scenes", stats["negative_scenes"])
            with col4:
                positive_ratio = stats["positive_scenes"] / stats["total_scenes"]
                st.metric("Positive Ratio", f"{positive_ratio:.1%}")
            
            # Sound type distribution
            st.markdown("#### 🎵 Sound Type Distribution")
            sound_dist = stats.get("sound_type_distribution", {})
            
            if sound_dist:
                # Create bar chart
                df_sounds = pd.DataFrame(
                    list(sound_dist.items()),
                    columns=["Sound Type", "Count"]
                ).sort_values("Count", ascending=False)
                
                fig_sounds = px.bar(
                    df_sounds.head(15),  # Show top 15
                    x="Count",
                    y="Sound Type",
                    orientation="h",
                    title="Most Common Sound Types",
                    color="Count",
                    color_continuous_scale="viridis"
                )
                fig_sounds.update_layout(height=500)
                st.plotly_chart(fig_sounds, use_container_width=True)
                
                # Highlight elephant counts
                elephant_count = sound_dist.get("elephant", 0)
                st.info(f"🐘 **Elephant sounds**: {elephant_count} total occurrences")
            
            # Height distribution
            st.markdown("#### 📏 Height Distribution")
            height_dist = stats.get("height_distribution", {})
            
            if height_dist:
                df_height = pd.DataFrame(
                    list(height_dist.items()),
                    columns=["Height Range", "Count"]
                )
                
                fig_height = px.pie(
                    df_height,
                    values="Count",
                    names="Height Range",
                    title="Sound Height Distribution"
                )
                st.plotly_chart(fig_height, use_container_width=True)
            
            # Distance distribution  
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("#### 📍 Distance Distribution")
                distance_dist = stats.get("distance_distribution", {})
                
                if distance_dist:
                    df_distance = pd.DataFrame(
                        list(distance_dist.items()),
                        columns=["Distance Range", "Count"]
                    )
                    
                    fig_distance = px.bar(
                        df_distance,
                        x="Distance Range",
                        y="Count",
                        title="Sound Distance Distribution"
                    )
                    fig_distance.update_xaxes(tickangle=45)
                    st.plotly_chart(fig_distance, use_container_width=True)
            
            with col2:
                st.markdown("#### 🌿 Ambient Types")
                ambient_dist = stats.get("ambient_type_distribution", {})
                
                if ambient_dist:
                    df_ambient = pd.DataFrame(
                        list(ambient_dist.items()),
                        columns=["Ambient Type", "Count"]
                    )
                    
                    fig_ambient = px.pie(
                        df_ambient,
                        values="Count",
                        names="Ambient Type",
                        title="Ambient Sound Distribution"
                    )
                    st.plotly_chart(fig_ambient, use_container_width=True)
            
        except Exception as e:
            st.error(f"Could not load statistics: {str(e)}")
    
    def _render_help_tab(self):
        """Render help and documentation"""
        st.subheader("ℹ️ Scene Generator Help")
        
        st.markdown("""
        ### 🎯 Purpose
        The Scene Generator creates large datasets of realistic acoustic scenes for training elephant detection algorithms.
        Each scene contains a mix of directional sounds (animals, machinery, human activity) and ambient background sounds.
        
        ### 📊 Key Features
        
        **Positive/Negative Sampling:**
        - Positive samples contain 1-5 elephant sounds with realistic group positioning
        - Negative samples contain no elephant sounds
        - Configurable ratio (e.g., 40% positive, 60% negative)
        
        **Physics-Based Positioning:**
        - Height constraints based on sound source type
        - Birds and helicopters at realistic altitudes (5-100m)
        - Ground animals at appropriate heights (0-4m)
        - Human activities at human level (1-2.5m)
        
        **Realistic Acoustic Modeling:**
        - Uses pyroomacoustics for 3D sound propagation
        - Forest-specific anechoic environment simulation
        - 4-microphone array geometry (ReSpeaker USB)
        - Proper distance attenuation and time delays
        
        ### 🎵 Sound Classifications
        
        **Directional Sounds** (positioned in 3D space):
        - **Animals**: elephant, lion, monkey, bird, frog, squirrel, etc.
        - **Machinery**: chainsaw, generator, helicopter, vehicle engine
        - **Human Activity**: speaking, clapping, footsteps, whistling
        - **Tools**: axe, handsaw, woodchop
        - **Other**: gunshot, firework, tree falling
        
        **Ambient Sounds** (background, no specific direction):
        - **Weather**: rain, thunderstorm, wind, waterdrops
        - **Environment**: forest base, fire, insects, bird chirping
        - **Water**: streams, flowing water
        - **Silence**: minimal background
        
        ### 📁 Output Format
        
        **JSONL Configuration File:**
        - One JSON scene per line
        - Complete metadata including ground truth labels
        - Sound positions, types, timing, and file paths
        
        **4-Channel Audio Output:**
        - `scene_name_mic_1.wav` through `scene_name_mic_4.wav`
        - 16kHz sampling rate, 30-second duration (configurable)
        - Realistic microphone array geometry
        - Ground truth metadata in JSON format
        
        ### 🚀 Usage Workflow
        
        1. **Configure Parameters**: Set number of scenes, positive ratio, sound counts
        2. **Generate Scenes**: Creates JSONL file with scene configurations
        3. **Review Statistics**: Analyze sound distributions and scene properties
        4. **Render Audio**: Convert configurations to 4-channel WAV files
        5. **Train Models**: Use generated data for machine learning training
        
        ### 💡 Tips
        
        - Start with smaller datasets (10-50 scenes) to test parameters
        - Use 40-60% positive ratio for balanced training data
        - Increase sound counts (20-40 directional) for more complex scenes
        - Monitor generation time - audio rendering is the slowest step
        - Generated scenes include elephant groups (1-5 individuals) for realism
        
        ### ⚠️ Performance Notes
        
        - Scene generation is fast (1000+ scenes per minute)
        - Audio rendering is slower (~1-5 scenes per minute depending on complexity)
        - Large datasets (1000+ scenes) may take hours to render
        - Monitor disk space - each scene generates ~4MB of audio files
        
        ### 📊 Quality Validation
        
        Generated scenes include complete ground truth metadata for validation:
        - Exact elephant positions and timing
        - DOA (Direction of Arrival) angles
        - Distance and elevation information
        - Signal-to-noise ratios
        - Sound type classifications
        """)


def render_scene_generator_page(dataset_manager: DatasetManager):
    """Main function to render the scene generator page"""
    ui = SceneGeneratorUI(dataset_manager)
    ui.render()