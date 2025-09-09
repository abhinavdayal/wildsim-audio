"""
Streamlit integration for ZODAS pipeline
Provides UI components and visualization for Direction of Arrival analysis
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
from typing import Dict, List, Optional
import json

from zodas import ZODASPipeline


def run_zodas_dataset_analysis_ui():
    """ZODAS analysis UI that works with generated datasets"""
    
    st.markdown("### 🎯 ZODAS Direction of Arrival Analysis")
    st.markdown("Analyze rendered audio from generated datasets to validate DOA estimation performance")
    
    # Dataset selection
    with st.expander("📁 Dataset Selection", expanded=True):
        st.markdown("Select a generated dataset with rendered audio files:")
        
        # Find available datasets
        outputs_dir = Path("outputs/generated_scenes")
        available_datasets = []
        
        if outputs_dir.exists():
            for dataset_dir in outputs_dir.iterdir():
                if dataset_dir.is_dir():
                    audio_dir = dataset_dir / "rendered_audio"
                    jsonl_files = list(dataset_dir.glob("*.jsonl"))
                    
                    if audio_dir.exists() and jsonl_files:
                        # Count available scenes
                        scene_dirs = [d for d in audio_dir.iterdir() if d.is_dir()]
                        available_datasets.append({
                            'name': dataset_dir.name,
                            'path': dataset_dir,
                            'audio_dir': audio_dir,
                            'jsonl_file': jsonl_files[0],
                            'scene_count': len(scene_dirs)
                        })
        
        if not available_datasets:
            st.warning("⚠️ No datasets with rendered audio found. Generate and render a dataset first.")
            st.info("💡 Go to Scene Generator → Generate scenes → Render audio")
            return
        
        # Dataset selection
        dataset_options = [f"{d['name']} ({d['scene_count']} scenes)" for d in available_datasets]
        selected_idx = st.selectbox(
            "Select Dataset", 
            range(len(dataset_options)),
            format_func=lambda i: dataset_options[i]
        )
        
        selected_dataset = available_datasets[selected_idx]
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"📁 Dataset: {selected_dataset['name']}")
            st.info(f"🎵 Scenes: {selected_dataset['scene_count']}")
        with col2:
            st.info(f"📄 Config: {selected_dataset['jsonl_file'].name}")
            st.info(f"🎯 Audio: {selected_dataset['audio_dir'].name}/")
    
    # Scene selection
    with st.expander("🎬 Scene Selection", expanded=True):
        scene_dirs = [d for d in selected_dataset['audio_dir'].iterdir() if d.is_dir()]
        scene_names = [d.name for d in sorted(scene_dirs)]
        
        if not scene_names:
            st.error("No rendered scenes found in the selected dataset")
            return
        
        # Scene selection options
        analysis_mode = st.radio(
            "Analysis Mode",
            ["Single Scene", "Multiple Scenes", "Full Dataset"],
            horizontal=True
        )
        
        selected_scenes = []
        if analysis_mode == "Single Scene":
            selected_scene = st.selectbox("Select Scene", scene_names)
            selected_scenes = [selected_scene]
        elif analysis_mode == "Multiple Scenes":
            selected_scenes = st.multiselect(
                "Select Scenes", 
                scene_names,
                default=scene_names[:5] if len(scene_names) >= 5 else scene_names
            )
        else:  # Full Dataset
            selected_scenes = scene_names
            st.info(f"Analyzing all {len(scene_names)} scenes in dataset")
    
    # ZODAS Configuration
    with st.expander("⚙️ ZODAS Configuration", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sample_rate = st.selectbox("Sample Rate", [16000, 44100, 48000], index=0)
            frame_size = st.selectbox("Frame Size", [256, 512, 1024], index=1)
        
        with col2:
            overlap = st.slider("Frame Overlap", 0.0, 0.8, 0.5, 0.1)
        
        with col3:
            st.markdown("**Microphone Array**")
            st.markdown("ReSpeaker USB 4-mic")
            st.markdown("6.4cm diameter circular array")
    
    # Analysis execution
    if st.button("🚀 Run ZODAS Analysis", type="primary"):
        if not selected_scenes:
            st.error("Please select at least one scene to analyze")
            return
        
        try:
            with st.spinner(f"Processing {len(selected_scenes)} scene(s) with ZODAS..."):
                # Load ground truth from JSONL
                ground_truth_data = load_ground_truth_from_jsonl(selected_dataset['jsonl_file'], selected_scenes)
                
                # Process each scene
                all_results = []
                progress_bar = st.progress(0)
                
                for i, scene_name in enumerate(selected_scenes):
                    scene_dir = selected_dataset['audio_dir'] / scene_name
                    
                    # Load 4-channel audio
                    mic_files = []
                    for mic_num in range(1, 5):
                        mic_file = scene_dir / f"mic_{mic_num}.wav"
                        if mic_file.exists():
                            mic_files.append(str(mic_file))
                        else:
                            st.error(f"Missing audio file: {mic_file}")
                            return
                    
                    # Get ground truth for this scene
                    scene_ground_truth = ground_truth_data.get(scene_name, [])
                    
                    # Run ZODAS
                    zodas = ZODASPipeline(
                        sample_rate=sample_rate,
                        frame_size=frame_size,
                        overlap=overlap
                    )
                    
                    results = zodas.process_audio_files(mic_files, scene_ground_truth)
                    results['scene_name'] = scene_name
                    all_results.append(results)
                    
                    progress_bar.progress((i + 1) / len(selected_scenes))
                
                # Store results in session state
                st.session_state.zodas_dataset_results = {
                    'dataset_name': selected_dataset['name'],
                    'results': all_results,
                    'config': {
                        'sample_rate': sample_rate,
                        'frame_size': frame_size,
                        'overlap': overlap
                    }
                }
                
                st.success(f"✅ ZODAS analysis completed for {len(selected_scenes)} scene(s)!")
        
        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")
            return
    
    # Display results if available
    if 'zodas_dataset_results' in st.session_state:
        display_zodas_dataset_results(st.session_state.zodas_dataset_results)


def load_ground_truth_from_jsonl(jsonl_file: Path, scene_names: List[str]) -> Dict:
    """Load ground truth data for selected scenes from JSONL file"""
    
    ground_truth_data = {}
    
    try:
        with open(jsonl_file, 'r') as f:
            for line in f:
                scene_config = json.loads(line)
                scene_name = scene_config['name']
                
                if scene_name in scene_names:
                    # Extract elephant positions (positive scenes)
                    elephant_sources = []
                    for sound in scene_config['sounds']:
                        if sound['sound_type'] == 'elephant':
                            elephant_sources.append({
                                'position': sound['position'],
                                'azimuth': sound.get('azimuth', 0),
                                'elevation': sound.get('elevation', 0)
                            })
                    
                    if elephant_sources:
                        ground_truth_data[scene_name] = elephant_sources
    
    except Exception as e:
        st.warning(f"Could not load ground truth: {e}")
    
    return ground_truth_data


def display_zodas_dataset_results(dataset_results: Dict):
    """Display ZODAS analysis results for dataset"""
    
    st.markdown("## 📊 ZODAS Dataset Analysis Results")
    
    results_list = dataset_results['results']
    dataset_name = dataset_results['dataset_name']
    config = dataset_results['config']
    
    # Summary metrics
    total_scenes = len(results_list)
    scenes_with_validation = len([r for r in results_list if 'validation_metrics' in r])
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Dataset", dataset_name)
    
    with col2:
        st.metric("Scenes Analyzed", total_scenes)
    
    with col3:
        st.metric("With Ground Truth", scenes_with_validation)
    
    with col4:
        st.metric("Frame Size", f"{config['frame_size']} samples")
    
    # Results tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Dataset Performance", 
        "🎬 Scene-by-Scene", 
        "📊 Aggregated Analysis", 
        "💾 Export Results"
    ])
    
    with tab1:
        display_dataset_performance(results_list)
    
    with tab2:
        display_scene_by_scene_results(results_list)
    
    with tab3:
        display_aggregated_analysis(results_list)
    
    with tab4:
        display_export_options(dataset_results)


def display_dataset_performance(results_list: List[Dict]):
    """Display overall dataset performance metrics"""
    
    st.subheader("Dataset Performance Overview")
    
    # Collect validation metrics
    validation_results = [r for r in results_list if 'validation_metrics' in r]
    
    if not validation_results:
        st.info("No ground truth validation available for this dataset")
        return
    
    # Aggregate metrics
    all_az_errors = []
    all_el_errors = []
    all_accuracies_10 = []
    
    for result in validation_results:
        vm = result['validation_metrics']
        all_az_errors.extend(vm['errors']['azimuth_errors'])
        all_el_errors.extend(vm['errors']['elevation_errors'])
        all_accuracies_10.append(vm['accuracy_metrics']['azimuth_accuracy_10deg'])
    
    # Performance summary
    col1, col2, col3 = st.columns(3)
    
    with col1:
        mean_az_error = np.mean(all_az_errors)
        std_az_error = np.std(all_az_errors)
        st.metric(
            "Mean Azimuth Error",
            f"{mean_az_error:.2f}°",
            f"±{std_az_error:.2f}°"
        )
    
    with col2:
        mean_el_error = np.mean(all_el_errors)
        std_el_error = np.std(all_el_errors)
        st.metric(
            "Mean Elevation Error",
            f"{mean_el_error:.2f}°",
            f"±{std_el_error:.2f}°"
        )
    
    with col3:
        mean_accuracy = np.mean(all_accuracies_10)
        st.metric(
            "Average Accuracy (≤10°)",
            f"{mean_accuracy:.1f}%"
        )
    
    # Error distribution plots
    col1, col2 = st.columns(2)
    
    with col1:
        fig_az = go.Figure()
        fig_az.add_trace(go.Histogram(
            x=all_az_errors,
            nbinsx=30,
            name="Azimuth Errors",
            marker_color='blue'
        ))
        fig_az.update_layout(
            title="Azimuth Error Distribution",
            xaxis_title="Error (degrees)",
            yaxis_title="Count"
        )
        st.plotly_chart(fig_az, use_container_width=True)
    
    with col2:
        fig_el = go.Figure()
        fig_el.add_trace(go.Histogram(
            x=all_el_errors,
            nbinsx=30,
            name="Elevation Errors",
            marker_color='green'
        ))
        fig_el.update_layout(
            title="Elevation Error Distribution",
            xaxis_title="Error (degrees)",
            yaxis_title="Count"
        )
        st.plotly_chart(fig_el, use_container_width=True)


def display_scene_by_scene_results(results_list: List[Dict]):
    """Display individual scene results"""
    
    st.subheader("Scene-by-Scene Analysis")
    
    # Scene selection
    scene_names = [r['scene_name'] for r in results_list]
    selected_scene = st.selectbox("Select Scene for Detailed Analysis", scene_names)
    
    # Find selected result
    selected_result = next(r for r in results_list if r['scene_name'] == selected_scene)
    
    # Display the result using existing functions
    display_zodas_results(selected_result, st.session_state.zodas_dataset_results['config'])


def display_aggregated_analysis(results_list: List[Dict]):
    """Display aggregated analysis across all scenes"""
    
    st.subheader("Aggregated Dataset Analysis")
    
    # Performance by scene type
    positive_scenes = [r for r in results_list if 'validation_metrics' in r]
    negative_scenes = [r for r in results_list if 'validation_metrics' not in r]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Scene Distribution**")
        st.write(f"Positive scenes (with elephants): {len(positive_scenes)}")
        st.write(f"Negative scenes (no elephants): {len(negative_scenes)}")
    
    with col2:
        st.markdown("**Processing Statistics**")
        total_frames = sum(r['n_frames_processed'] for r in results_list)
        avg_duration = np.mean([r['audio_duration'] for r in results_list])
        st.write(f"Total frames processed: {total_frames:,}")
        st.write(f"Average scene duration: {avg_duration:.1f}s")
    
    # Accuracy trends
    if positive_scenes:
        st.subheader("Accuracy Trends")
        
        scene_names = [r['scene_name'] for r in positive_scenes]
        accuracies = [r['validation_metrics']['accuracy_metrics']['azimuth_accuracy_10deg'] 
                     for r in positive_scenes]
        
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=list(range(len(scene_names))),
            y=accuracies,
            mode='lines+markers',
            name='Accuracy (≤10°)',
            text=scene_names,
            hovertemplate='%{text}<br>Accuracy: %{y:.1f}%<extra></extra>'
        ))
        
        fig_trend.update_layout(
            title="DOA Accuracy Across Scenes",
            xaxis_title="Scene Index",
            yaxis_title="Accuracy (%)",
            height=400
        )
        
        st.plotly_chart(fig_trend, use_container_width=True)


def display_export_options(dataset_results: Dict):
    """Display export options for dataset results"""
    
    st.subheader("Export Dataset Analysis")
    
    dataset_name = dataset_results['dataset_name']
    results_list = dataset_results['results']
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Export summary CSV
        summary_data = []
        for result in results_list:
            row = {
                'scene_name': result['scene_name'],
                'duration': result['audio_duration'],
                'frames_processed': result['n_frames_processed']
            }
            
            if 'validation_metrics' in result:
                vm = result['validation_metrics']
                row.update({
                    'mean_azimuth_error': vm['errors']['mean_azimuth_error'],
                    'mean_elevation_error': vm['errors']['mean_elevation_error'],
                    'azimuth_accuracy_10deg': vm['accuracy_metrics']['azimuth_accuracy_10deg'],
                    'elevation_accuracy_10deg': vm['accuracy_metrics']['elevation_accuracy_10deg']
                })
            
            summary_data.append(row)
        
        df_summary = pd.DataFrame(summary_data)
        csv_summary = df_summary.to_csv(index=False)
        
        st.download_button(
            label="📊 Download Summary CSV",
            data=csv_summary,
            file_name=f"{dataset_name}_zodas_summary.csv",
            mime="text/csv"
        )
    
    with col2:
        # Export full results JSON - need to convert tuple keys to strings for JSON serialization
        def convert_tuple_keys(obj):
            """Recursively convert tuple keys to strings for JSON serialization"""
            if isinstance(obj, dict):
                new_dict = {}
                for key, value in obj.items():
                    # Convert tuple keys to string representation
                    if isinstance(key, tuple):
                        new_key = str(key)
                    else:
                        new_key = key
                    new_dict[new_key] = convert_tuple_keys(value)
                return new_dict
            elif isinstance(obj, list):
                return [convert_tuple_keys(item) for item in obj]
            else:
                return obj
        
        # Convert the full results to handle tuple keys
        serializable_results = convert_tuple_keys(results_list)
        
        export_data = {
            'dataset_name': dataset_name,
            'analysis_config': dataset_results['config'],
            'total_scenes': len(results_list),
            'results_summary': summary_data,
            'full_results': serializable_results
        }
        
        json_data = json.dumps(export_data, indent=2, default=str)
        
        st.download_button(
            label="📄 Download Full Results JSON",
            data=json_data,
            file_name=f"{dataset_name}_zodas_full_results.json",
            mime="application/json"
        )


def run_zodas_analysis_ui():
    """Main ZODAS analysis UI component for Streamlit"""
    
    st.markdown("### 🎯 ZODAS Direction of Arrival Analysis")
    st.markdown("Process 4-channel microphone array recordings to estimate sound source directions")
    
    # Configuration section
    with st.expander("⚙️ Configuration", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sample_rate = st.selectbox("Sample Rate", [16000, 44100, 48000], index=0)
            frame_size = st.selectbox("Frame Size", [256, 512, 1024], index=1)
        
        with col2:
            overlap = st.slider("Frame Overlap", 0.0, 0.8, 0.5, 0.1)
        
        with col3:
            st.markdown("**Microphone Array**")
            st.markdown("ReSpeaker USB 4-mic")
            st.markdown("6.4cm diameter circular array")
    
    # File upload section
    st.markdown("### 📁 Audio Files")
    st.markdown("Upload 4 mono WAV files (one per microphone)")
    
    col1, col2, col3, col4 = st.columns(4)
    
    uploaded_files = {}
    with col1:
        uploaded_files['mic1'] = st.file_uploader("Mic 1 (Left)", type=['wav'], key='mic1')
    with col2:
        uploaded_files['mic2'] = st.file_uploader("Mic 2 (Back)", type=['wav'], key='mic2')
    with col3:
        uploaded_files['mic3'] = st.file_uploader("Mic 3 (Right)", type=['wav'], key='mic3')
    with col4:
        uploaded_files['mic4'] = st.file_uploader("Mic 4 (Front)", type=['wav'], key='mic4')
    
    # Ground truth input (optional)
    with st.expander("🎯 Ground Truth (Optional)", expanded=False):
        st.markdown("Provide known source position for validation")
        
        col1, col2 = st.columns(2)
        with col1:
            gt_azimuth = st.number_input("Azimuth (degrees)", -180.0, 180.0, 0.0, 1.0)
            gt_elevation = st.number_input("Elevation (degrees)", -90.0, 90.0, 0.0, 1.0)
        
        with col2:
            gt_x = st.number_input("X position (m)", -100.0, 100.0, 10.0, 1.0)
            gt_y = st.number_input("Y position (m)", -100.0, 100.0, 0.0, 1.0)
            gt_z = st.number_input("Z position (m)", 0.0, 50.0, 2.0, 0.5)
        
        use_ground_truth = st.checkbox("Enable validation against ground truth")
    
    # Process button
    if st.button("🚀 Run ZODAS Analysis", type="primary"):
        if not all(uploaded_files.values()):
            st.error("Please upload all 4 microphone files")
            return
        
        try:
            # Save uploaded files temporarily
            temp_files = []
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_dir = Path(temp_dir)
                
                for i, (mic_name, uploaded_file) in enumerate(uploaded_files.items()):
                    if uploaded_file is not None:
                        temp_file = temp_dir / f"{mic_name}.wav"
                        with open(temp_file, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        temp_files.append(str(temp_file))
                
                # Set up ground truth if provided
                ground_truth = None
                if use_ground_truth:
                    ground_truth = [{
                        'position': [gt_x, gt_y, gt_z],
                        'azimuth': gt_azimuth,
                        'elevation': gt_elevation
                    }]
                
                # Run ZODAS pipeline
                with st.spinner("Processing audio with ZODAS..."):
                    zodas = ZODASPipeline(
                        sample_rate=sample_rate,
                        frame_size=frame_size,
                        overlap=overlap
                    )
                    
                    results = zodas.process_audio_files(temp_files, ground_truth)
                
                # Store results in session state
                st.session_state.zodas_results = results
                st.session_state.zodas_config = {
                    'sample_rate': sample_rate,
                    'frame_size': frame_size,
                    'overlap': overlap
                }
                
                st.success("✅ ZODAS analysis completed!")
        
        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")
            return
    
    # Display results if available
    if 'zodas_results' in st.session_state:
        display_zodas_results(st.session_state.zodas_results, st.session_state.zodas_config)


def display_zodas_results(results: Dict, config: Dict):
    """Display ZODAS analysis results with interactive visualizations"""
    
    st.markdown("## 📊 ZODAS Analysis Results")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Duration", f"{results['audio_duration']:.2f}s")
    
    with col2:
        st.metric("Frames Processed", results['n_frames_processed'])
    
    with col3:
        st.metric("Frame Size", f"{config['frame_size']} samples")
    
    with col4:
        has_validation = 'validation_metrics' in results
        st.metric("Validation", "✅ Available" if has_validation else "❌ No GT")
    
    # Main results tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 DOA Estimation", 
        "📈 Validation Metrics", 
        "📊 Signal Analysis", 
        "🗂️ Detailed Data"
    ])
    
    with tab1:
        display_doa_estimation(results)
    
    with tab2:
        if has_validation:
            display_validation_metrics(results['validation_metrics'])
        else:
            st.info("No ground truth provided - validation metrics not available")
    
    with tab3:
        display_signal_analysis(results)
    
    with tab4:
        display_detailed_data(results, config)


def display_doa_estimation(results: Dict):
    """Display DOA estimation results"""
    
    st.subheader("Direction of Arrival Estimates")
    
    # Extract data
    frame_results = results['frame_results']
    timestamps = [r['timestamp'] for r in frame_results]
    azimuths = [r['doa_azimuth'] for r in frame_results]
    elevations = [r['doa_elevation'] for r in frame_results]
    
    # Time series plots
    fig_time = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Azimuth Over Time', 'Elevation Over Time'),
        vertical_spacing=0.1
    )
    
    # Azimuth plot
    fig_time.add_trace(
        go.Scatter(x=timestamps, y=azimuths, mode='lines+markers', 
                  name='Estimated Azimuth', line=dict(color='blue')),
        row=1, col=1
    )
    
    # Add ground truth if available
    if 'validation_metrics' in results:
        gt_az = results['validation_metrics']['ground_truth']['azimuth']
        fig_time.add_hline(y=gt_az, line_dash="dash", line_color="red", 
                          annotation_text="Ground Truth", row=1, col=1)
    
    # Elevation plot
    fig_time.add_trace(
        go.Scatter(x=timestamps, y=elevations, mode='lines+markers',
                  name='Estimated Elevation', line=dict(color='green')),
        row=2, col=1
    )
    
    if 'validation_metrics' in results:
        gt_el = results['validation_metrics']['ground_truth']['elevation']
        fig_time.add_hline(y=gt_el, line_dash="dash", line_color="red",
                          annotation_text="Ground Truth", row=2, col=1)
    
    fig_time.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig_time.update_yaxes(title_text="Azimuth (°)", row=1, col=1)
    fig_time.update_yaxes(title_text="Elevation (°)", row=2, col=1)
    fig_time.update_layout(height=600, title="DOA Estimates Over Time")
    
    st.plotly_chart(fig_time, use_container_width=True)
    
    # 2D trajectory plot
    st.subheader("DOA Trajectory (2D View)")
    
    fig_2d = go.Figure()
    
    # Estimated trajectory
    fig_2d.add_trace(go.Scatter(
        x=azimuths, y=elevations,
        mode='markers+lines',
        marker=dict(
            size=8,
            color=timestamps,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Time (s)")
        ),
        line=dict(width=2, color='rgba(0,0,255,0.3)'),
        name='Estimated DOA',
        text=[f't={t:.2f}s' for t in timestamps],
        hovertemplate='Az: %{x:.1f}°<br>El: %{y:.1f}°<br>%{text}<extra></extra>'
    ))
    
    # Ground truth point
    if 'validation_metrics' in results:
        gt_az = results['validation_metrics']['ground_truth']['azimuth']
        gt_el = results['validation_metrics']['ground_truth']['elevation']
        fig_2d.add_trace(go.Scatter(
            x=[gt_az], y=[gt_el],
            mode='markers',
            marker=dict(size=15, color='red', symbol='star'),
            name='Ground Truth',
            hovertemplate='Ground Truth<br>Az: %{x:.1f}°<br>El: %{y:.1f}°<extra></extra>'
        ))
    
    fig_2d.update_layout(
        title="DOA Trajectory in Azimuth-Elevation Space",
        xaxis_title="Azimuth (degrees)",
        yaxis_title="Elevation (degrees)",
        width=800, height=600
    )
    
    st.plotly_chart(fig_2d, use_container_width=True)


def display_validation_metrics(validation_metrics: Dict):
    """Display validation metrics and error analysis"""
    
    st.subheader("Validation Metrics")
    
    vm = validation_metrics
    
    # Summary statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Mean Azimuth Error",
            f"{vm['errors']['mean_azimuth_error']:.2f}°",
            f"±{vm['errors']['std_azimuth_error']:.2f}°"
        )
    
    with col2:
        st.metric(
            "Mean Elevation Error", 
            f"{vm['errors']['mean_elevation_error']:.2f}°",
            f"±{vm['errors']['std_elevation_error']:.2f}°"
        )
    
    with col3:
        st.metric(
            "Accuracy (≤10°)",
            f"Az: {vm['accuracy_metrics']['azimuth_accuracy_10deg']:.1f}%",
            f"El: {vm['accuracy_metrics']['elevation_accuracy_10deg']:.1f}%"
        )
    
    # Error time series
    st.subheader("Error Over Time")
    
    timestamps = vm['estimates']['timestamps']
    az_errors = vm['errors']['azimuth_errors']
    el_errors = vm['errors']['elevation_errors']
    
    fig_errors = go.Figure()
    
    fig_errors.add_trace(go.Scatter(
        x=timestamps, y=az_errors,
        mode='lines+markers',
        name='Azimuth Error',
        line=dict(color='blue')
    ))
    
    fig_errors.add_trace(go.Scatter(
        x=timestamps, y=el_errors,
        mode='lines+markers',
        name='Elevation Error',
        line=dict(color='green')
    ))
    
    fig_errors.update_layout(
        title="Estimation Errors Over Time",
        xaxis_title="Time (s)",
        yaxis_title="Error (degrees)",
        height=400
    )
    
    st.plotly_chart(fig_errors, use_container_width=True)
    
    # Accuracy breakdown
    st.subheader("Accuracy Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Accuracy by tolerance
        tolerances = ['5°', '10°', '15°']
        az_accuracy = [
            vm['accuracy_metrics']['azimuth_accuracy_5deg'],
            vm['accuracy_metrics']['azimuth_accuracy_10deg'],
            vm['accuracy_metrics']['azimuth_accuracy_15deg']
        ]
        el_accuracy = [
            vm['accuracy_metrics']['elevation_accuracy_5deg'],
            vm['accuracy_metrics']['elevation_accuracy_10deg'],
            0  # 15° not computed for elevation
        ]
        
        fig_acc = go.Figure()
        fig_acc.add_trace(go.Bar(
            name='Azimuth',
            x=tolerances,
            y=az_accuracy,
            marker_color='blue',
            text=[f'{v:.1f}%' for v in az_accuracy],
            textposition='auto'
        ))
        
        fig_acc.add_trace(go.Bar(
            name='Elevation',
            x=tolerances[:2],  # Only 5° and 10°
            y=el_accuracy[:2],
            marker_color='green',
            text=[f'{v:.1f}%' for v in el_accuracy[:2]],
            textposition='auto'
        ))
        
        fig_acc.update_layout(
            title="Accuracy within Tolerance",
            xaxis_title="Error Tolerance",
            yaxis_title="Accuracy (%)",
            barmode='group',
            height=400
        )
        
        st.plotly_chart(fig_acc, use_container_width=True)
    
    with col2:
        # Error distribution
        fig_hist = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Azimuth Error Distribution', 'Elevation Error Distribution')
        )
        
        fig_hist.add_trace(
            go.Histogram(x=az_errors, nbinsx=20, name='Azimuth', marker_color='blue'),
            row=1, col=1
        )
        
        fig_hist.add_trace(
            go.Histogram(x=el_errors, nbinsx=20, name='Elevation', marker_color='green'),
            row=2, col=1
        )
        
        fig_hist.update_layout(
            title="Error Distributions",
            height=400,
            showlegend=False
        )
        fig_hist.update_xaxes(title_text="Error (degrees)")
        fig_hist.update_yaxes(title_text="Count")
        
        st.plotly_chart(fig_hist, use_container_width=True)
    
    # Detailed statistics table
    st.subheader("Detailed Statistics")
    
    stats_data = {
        'Metric': [
            'Mean Error', 'Std Error', 'Median Error', 'Max Error',
            'Accuracy ≤5°', 'Accuracy ≤10°', 'Temporal Consistency'
        ],
        'Azimuth': [
            f"{vm['errors']['mean_azimuth_error']:.2f}°",
            f"{vm['errors']['std_azimuth_error']:.2f}°",
            f"{vm['errors']['median_azimuth_error']:.2f}°",
            f"{vm['errors']['max_azimuth_error']:.2f}°",
            f"{vm['accuracy_metrics']['azimuth_accuracy_5deg']:.1f}%",
            f"{vm['accuracy_metrics']['azimuth_accuracy_10deg']:.1f}%",
            f"{vm['stability_metrics']['temporal_consistency']:.2f}°"
        ],
        'Elevation': [
            f"{vm['errors']['mean_elevation_error']:.2f}°",
            f"{vm['errors']['std_elevation_error']:.2f}°",
            f"{vm['errors']['median_elevation_error']:.2f}°",
            f"{vm['errors']['max_elevation_error']:.2f}°",
            f"{vm['accuracy_metrics']['elevation_accuracy_5deg']:.1f}%",
            f"{vm['accuracy_metrics']['elevation_accuracy_10deg']:.1f}%",
            "-"
        ]
    }
    
    df_stats = pd.DataFrame(stats_data)
    st.dataframe(df_stats, use_container_width=True)


def display_signal_analysis(results: Dict):
    """Display signal processing analysis"""
    
    st.subheader("Signal Processing Analysis")
    
    # Spatial spectrogram
    if results['spatial_spectrogram'].size > 0:
        st.subheader("Spatial Spectrogram")
        
        fig_ss = go.Figure(data=go.Heatmap(
            z=results['spatial_spectrogram'].T,
            colorscale='Viridis',
            showscale=True
        ))
        
        fig_ss.update_layout(
            title="Spatial Spectrogram Matrix",
            xaxis_title="Frame Index",
            yaxis_title="Frequency Bin",
            height=400
        )
        
        st.plotly_chart(fig_ss, use_container_width=True)
    
    # Average spectrum
    st.subheader("Average Magnitude Spectrum")
    
    avg_spectrum = np.mean([np.mean(r['magnitude_spectrum'], axis=1) 
                           for r in results['frame_results']], axis=0)
    
    fig_spec = go.Figure()
    fig_spec.add_trace(go.Scatter(
        x=results['freqs'],
        y=avg_spectrum,
        mode='lines',
        name='Average Spectrum'
    ))
    
    fig_spec.update_layout(
        title="Average Magnitude Spectrum",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Magnitude",
        height=400
    )
    
    st.plotly_chart(fig_spec, use_container_width=True)
    
    # Spectral peaks
    st.subheader("Spectral Peaks Over Time")
    
    peak_data = []
    for r in results['frame_results']:
        for freq, mag in r['spectral_peaks'][:5]:
            peak_data.append({
                'Time': r['timestamp'],
                'Frequency': freq,
                'Magnitude': mag
            })
    
    if peak_data:
        df_peaks = pd.DataFrame(peak_data)
        
        fig_peaks = px.scatter(
            df_peaks, x='Time', y='Frequency', size='Magnitude',
            title="Spectral Peaks Over Time",
            labels={'Time': 'Time (s)', 'Frequency': 'Frequency (Hz)'}
        )
        
        st.plotly_chart(fig_peaks, use_container_width=True)


def display_detailed_data(results: Dict, config: Dict):
    """Display detailed raw data and download options"""
    
    st.subheader("Detailed Data")
    
    # DOA estimates table
    st.subheader("DOA Estimates by Frame")
    
    frame_data = []
    for r in results['frame_results']:
        frame_data.append({
            'Frame': r['frame_index'],
            'Time (s)': f"{r['timestamp']:.3f}",
            'Azimuth (°)': f"{r['doa_azimuth']:.2f}",
            'Elevation (°)': f"{r['doa_elevation']:.2f}"
        })
    
    df_frames = pd.DataFrame(frame_data)
    st.dataframe(df_frames, use_container_width=True)
    
    # Download options
    st.subheader("Export Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Export DOA estimates as CSV
        csv_data = df_frames.to_csv(index=False)
        st.download_button(
            label="📊 Download DOA Estimates (CSV)",
            data=csv_data,
            file_name="zodas_doa_estimates.csv",
            mime="text/csv"
        )
    
    with col2:
        # Export full results as JSON
        results_json = json.dumps({
            'config': config,
            'summary': {
                'duration': results['audio_duration'],
                'n_frames': results['n_frames_processed']
            },
            'doa_estimates': frame_data,
            'validation_metrics': results.get('validation_metrics', {})
        }, indent=2)
        
        st.download_button(
            label="📄 Download Full Results (JSON)",
            data=results_json,
            file_name="zodas_analysis_results.json",
            mime="application/json"
        )
    
    # Validation report
    if 'validation_metrics' in results:
        st.subheader("Validation Report")
        
        # Import the report generation function from zodas.py
        from zodas import ZODASPipeline
        zodas_temp = ZODASPipeline()
        report = zodas_temp.generate_validation_report(results)
        
        st.text_area("Validation Report", report, height=400)
        
        st.download_button(
            label="📝 Download Validation Report",
            data=report,
            file_name="zodas_validation_report.txt",
            mime="text/plain"
        )


def zodas_sidebar_help():
    """Add ZODAS help information to sidebar"""
    
    with st.sidebar:
        st.markdown("### 🎯 ZODAS Help")
        
        with st.expander("What is ZODAS?"):
            st.markdown("""
            **ZODAS** (Zone Of Detection And Signal processing) is a Direction of Arrival (DOA) 
            estimation pipeline designed for microphone arrays.
            
            It uses:
            - **GCC-PHAT**: Generalized Cross-Correlation with Phase Transform
            - **TDOA**: Time Difference of Arrival measurements
            - **Least Squares**: For 2D DOA estimation
            """)
        
        with st.expander("File Requirements"):
            st.markdown("""
            Upload **4 mono WAV files**:
            - Same sample rate (16kHz recommended)
            - Same duration
            - Synchronized recordings
            - One file per microphone
            """)
        
        with st.expander("Validation Metrics"):
            st.markdown("""
            When ground truth is provided:
            - **Mean/Std Error**: Average estimation error
            - **Accuracy**: Percentage within tolerance
            - **Temporal Consistency**: Stability over time
            - **Confidence Intervals**: Statistical reliability
            """)
