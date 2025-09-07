"""
Streamlit Web Interface for Wildlife Acoustic Scene Simulation
Provides interactive configuration, simulation, and visualization
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import json
from pathlib import Path
import tempfile
import zipfile
from datetime import datetime
import io
import time

# Import our modules
from models import *
from dataset_manager import DatasetManager, SmartAudioSelector
from auth_helpers import *
from scene_creator import render_scene_creator_tab
from scene_generator_ui import render_scene_generator_page
import pyroomacoustics as pra

# Page configuration
st.set_page_config(
    page_title="Wildlife Acoustic Simulation",
    page_icon="🐘",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #4682B4;
        border-bottom: 2px solid #4682B4;
        padding-bottom: 0.5rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2E8B57;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 0.75rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 0.75rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 0.75rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class SimulationSession:
    """Manages simulation session state"""
    
    def __init__(self):
        if 'simulation_results' not in st.session_state:
            st.session_state.simulation_results = {}
        if 'dataset_manager' not in st.session_state:
            st.session_state.dataset_manager = None
        if 'current_config' not in st.session_state:
            st.session_state.current_config = None

def dataset_manager_page():
    """Dataset management page"""
    
    st.markdown('<h2 class="section-header">📊 Dataset Manager</h2>', unsafe_allow_html=True)
    
    # Initialize dataset manager
    if st.session_state.dataset_manager is None:
        with st.spinner("Initializing dataset manager..."):
            st.session_state.dataset_manager = DatasetManager()
    
    dataset_manager = st.session_state.dataset_manager
    
    # Tabs for different dataset operations
    tab1, tab2, tab3, tab4 = st.tabs(["📥 Add Dataset", "📋 Dataset Summary", "🔍 Browse Files", "⚙️ Settings"])
    
    with tab1:
        st.subheader("Add New Dataset")
        
        # Dataset configuration form with dynamic fields
        dataset_name = st.text_input("Dataset Name", "My Dataset")
        
        # Improved source type selection with descriptions
        source_options = [
            ("local", "📁 Local - Existing files on your computer"),
            ("kaggle", "🏆 Kaggle - Download from Kaggle datasets"),
            ("google_drive", "☁️ Google Drive - Download from Google Drive folder"),
            ("url", "🌐 URL - Download from direct web link")
        ]
        
        source_type = st.selectbox(
            "Source Type", 
            options=[opt[0] for opt in source_options],
            format_func=lambda x: next(opt[1] for opt in source_options if opt[0] == x),
            help="Choose where your audio dataset is located"
        )
        
        path_or_id = ""  # Initialize
        
        # Dynamic form based on source type
        if source_type == "kaggle":
            st.markdown("### 🔑 Kaggle Configuration")
            
            # Check for Kaggle authentication
            kaggle_setup = check_kaggle_auth()
            if not kaggle_setup['authenticated']:
                st.error("❌ Kaggle authentication not configured")
                with st.expander("🔧 Setup Kaggle Authentication"):
                    st.markdown("""
                    **Steps to configure Kaggle API:**
                    1. Go to [Kaggle Account Settings](https://www.kaggle.com/settings/account)
                    2. Scroll to "API" section and click "Create New Token"
                    3. Download the `kaggle.json` file
                    4. Place it in one of these locations:
                       - `~/.kaggle/kaggle.json` (recommended)
                       - Upload using the button below
                    """)
                    
                    uploaded_kaggle = st.file_uploader(
                        "Upload kaggle.json", 
                        type="json",
                        help="Upload your Kaggle API credentials file"
                    )
                    
                    if uploaded_kaggle:
                        if setup_kaggle_credentials(uploaded_kaggle):
                            st.success("✅ Kaggle credentials configured successfully!")
                            st.rerun()
                        else:
                            st.error("❌ Failed to configure Kaggle credentials")
            else:
                st.success(f"✅ Kaggle authenticated as: {kaggle_setup['username']}")
            
            col1, col2 = st.columns(2)
            with col1:
                path_or_id = st.text_input(
                    "Kaggle Dataset ID", 
                    placeholder="username/dataset-name",
                    help="Enter the dataset ID from Kaggle URL (e.g., 'username/dataset-name')"
                )
                
                dataset_type = st.radio(
                    "Dataset Type",
                    ["Dataset", "Competition"],
                    help="Choose whether this is a regular dataset or competition data"
                )
            
            with col2:
                if st.button("🔍 Browse Kaggle Datasets"):
                    st.info("Visit [Kaggle Datasets](https://www.kaggle.com/datasets) to find datasets")
                
                if path_or_id and kaggle_setup['authenticated']:
                    if st.button("📊 Preview Dataset Info"):
                        try:
                            dataset_info = get_kaggle_dataset_info(path_or_id, dataset_type)
                            if dataset_info:
                                st.json(dataset_info)
                        except Exception as e:
                            st.error(f"Failed to fetch dataset info: {e}")
        
        elif source_type == "google_drive":
            st.markdown("### 🔑 Google Drive Configuration")
            
            # Check for Google Drive authentication
            gdrive_setup = check_gdrive_auth()
            if not gdrive_setup['authenticated']:
                st.error("❌ Google Drive authentication not configured")
                with st.expander("🔧 Setup Google Drive Authentication"):
                    st.markdown("""
                    **Steps to configure Google Drive API:**
                    1. Go to [Google Cloud Console](https://console.cloud.google.com/)
                    2. Create a new project or select existing one
                    3. Enable Google Drive API
                    4. Create credentials (OAuth 2.0 Client ID)
                    5. Download the credentials JSON file
                    6. Upload it using the button below
                    """)
                    
                    uploaded_gdrive = st.file_uploader(
                        "Upload credentials.json", 
                        type="json",
                        help="Upload your Google Drive API credentials file"
                    )
                    
                    if uploaded_gdrive:
                        if setup_gdrive_credentials(uploaded_gdrive):
                            st.success("✅ Google Drive credentials configured successfully!")
                            st.rerun()
                        else:
                            st.error("❌ Failed to configure Google Drive credentials")
            else:
                st.success("✅ Google Drive authenticated")
            
            col1, col2 = st.columns(2)
            with col1:
                path_or_id = st.text_input(
                    "Google Drive Folder ID",
                    placeholder="1A2B3C4D5E6F7G8H9I0J",
                    help="Extract folder ID from Google Drive URL: https://drive.google.com/drive/folders/FOLDER_ID"
                )
                
                folder_access = st.radio(
                    "Folder Access",
                    ["Public/Shared with me", "My Drive"],
                    help="Specify if folder is shared with you or in your personal drive"
                )
            
            with col2:
                if path_or_id and gdrive_setup['authenticated']:
                    if st.button("📁 Preview Folder Contents"):
                        try:
                            folder_info = get_gdrive_folder_info(path_or_id)
                            if folder_info:
                                st.write(f"**Folder:** {folder_info['name']}")
                                st.write(f"**Files:** {len(folder_info['files'])}")
                                with st.expander("View files"):
                                    for file in folder_info['files'][:20]:  # Show first 20
                                        st.write(f"- {file['name']} ({file['size']})")
                        except Exception as e:
                            st.error(f"Failed to fetch folder info: {e}")
        
        elif source_type == "url":
            st.markdown("### 🌐 URL Download Configuration")
            col1, col2 = st.columns(2)
            with col1:
                path_or_id = st.text_input(
                    "Download URL",
                    placeholder="https://example.com/dataset.zip",
                    help="Enter the direct download URL for the dataset archive"
                )
                
                url_auth_required = st.checkbox(
                    "Requires Authentication",
                    help="Check if the URL requires authentication headers"
                )
            
            with col2:
                if url_auth_required:
                    auth_header = st.text_input(
                        "Authorization Header",
                        type="password",
                        placeholder="Bearer token123...",
                        help="Enter authorization header if required"
                    )
                
                if path_or_id:
                    if st.button("🔍 Test URL"):
                        if test_url_accessibility(path_or_id):
                            st.success("✅ URL is accessible")
                        else:
                            st.error("❌ URL is not accessible")
        
        else:  # local
            st.markdown("### 📁 Local Path Configuration")
            st.info("💡 Point to an existing folder with audio files. Files will be cataloged but remain in their original location.")
            
            col1, col2 = st.columns(2)
            with col1:
                path_or_id = st.text_input(
                    "Local Directory Path", 
                    placeholder="./audio_cache/tiny_test",
                    help="Enter path to local directory containing audio files (absolute or relative path)"
                )
            
            with col2:
                if st.button("� Check Path"):
                    if path_or_id and os.path.exists(path_or_id):
                        # Count audio files specifically
                        audio_exts = ['.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aiff']
                        audio_files = []
                        for ext in audio_exts:
                            audio_files.extend(list(Path(path_or_id).rglob(f"*{ext}")))
                        
                        total_files = len([f for f in Path(path_or_id).rglob("*") if f.is_file()])
                        st.success(f"✅ Found {len(audio_files)} audio files ({total_files} total files)")
                        
                        if audio_files:
                            with st.expander("📂 Sample audio files"):
                                for audio_file in audio_files[:5]:  # Show first 5
                                    st.write(f"🎵 {audio_file.name}")
                    else:
                        st.error("❌ Path does not exist")
                        st.caption("Make sure the path exists and you have read permissions")
        
        # Common configuration options
        st.markdown("### ⚙️ Dataset Configuration")
        col1, col2 = st.columns(2)
        
        with col1:
            audio_extensions = st.multiselect(
                "Audio Extensions", 
                [".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aiff"],
                default=[".wav", ".mp3"],
                help="Select which audio file types to include"
            )
            
            metadata_file = st.text_input(
                "Metadata File (optional)",
                placeholder="metadata.csv",
                help="Optional CSV file with audio metadata (filename, duration, labels, etc.)"
            )
        
        with col2:
            # Only show cache directory for non-local datasets
            if source_type != "local":
                cache_dir = st.text_input(
                    "Cache Directory",
                    value=f"./audio_cache/{dataset_name.lower().replace(' ', '_')}",
                    help="Directory where downloaded dataset will be cached locally"
                )
                
                overwrite_cache = st.checkbox(
                    "Overwrite existing cache",
                    help="Check to re-download even if dataset exists in cache"
                )
            else:
                # For local datasets, we don't need cache settings
                st.info("ℹ️ Local datasets are cataloged in place - no caching needed")
                cache_dir = "./audio_cache"  # Default cache for catalog storage
                overwrite_cache = False
        
        # Check if dataset already exists
        dataset_exists = False
        existing_file_count = 0
        
        if dataset_name and dataset_manager:
            try:
                # Check if dataset already exists in catalog
                if dataset_name in dataset_manager.catalog.catalog["datasets"]:
                    dataset_info = dataset_manager.catalog.catalog["datasets"][dataset_name]
                    existing_file_count = dataset_info.get("file_count", 0)
                    dataset_exists = existing_file_count > 0
            except:
                pass
        
        # Submit button with validation and existing dataset check
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            # Show different button states based on dataset existence
            if dataset_exists and not overwrite_cache:
                st.success(f"✅ Dataset '{dataset_name}' already set up with {existing_file_count} files")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    submit = st.button(
                        "🔄 Re-setup Dataset", 
                        type="secondary",
                        width='stretch',
                        help="Re-process this dataset (will overwrite existing catalog)"
                    )
                with col_b:
                    if st.button("📊 View Summary", type="primary", width='stretch'):
                        st.info("💡 Go to the 'Dataset Summary' tab above to view dataset details")
            else:
                # Standard setup button
                button_disabled = not path_or_id or not dataset_name
                submit = st.button(
                    "🔄 Setup Dataset", 
                    type="primary",
                    width='stretch',
                    disabled=button_disabled,
                    help="Download and catalog this dataset" if not button_disabled else "Please fill in all required fields"
                )
            
            if submit:
                try:
                    config = DatasetConfig(
                        dataset_name=dataset_name,
                        source_type=source_type,
                        path_or_id=path_or_id,
                        audio_extensions=audio_extensions,
                        metadata_file=metadata_file if metadata_file else None,
                        cache_dir=cache_dir
                    )
                    
                    # Initialize progress tracking in session state
                    if 'dataset_setup_progress' not in st.session_state:
                        st.session_state.dataset_setup_progress = None
                    
                    # Create a progress container
                    progress_container = st.container()
                    
                    # Start dataset setup with progress tracking
                    try:
                        with progress_container:
                            st.info(f"🔄 Setting up {dataset_name}...")
                            progress_bar = st.progress(0.0)
                            status_text = st.empty()
                            
                            # Download phase
                            status_text.text("📥 Downloading dataset...")
                            progress_bar.progress(0.1)
                            
                            # Setup dataset (download only, skip cataloging for now)
                            success = dataset_manager.setup_dataset_download_only(config)
                            
                            if success:
                                progress_bar.progress(0.5)
                                status_text.text("📊 Cataloging audio files...")
                                
                                # Get file count for progress calculation
                                dataset_path = Path(config.cache_dir)
                                audio_files = []
                                for ext in config.audio_extensions:
                                    audio_files.extend(list(dataset_path.rglob(f"*{ext}")))
                                
                                total_files = len(audio_files)
                                status_text.text(f"📊 Processing {total_files} audio files...")
                                
                                # Process files in chunks to show progress
                                chunk_size = max(50, total_files // 20)  # Process in ~20 chunks
                                processed = 0
                                
                                for i in range(0, total_files, chunk_size):
                                    chunk_files = audio_files[i:i+chunk_size]
                                    
                                    # Process chunk
                                    dataset_manager.catalog.process_audio_chunk(chunk_files, config)
                                    
                                    processed += len(chunk_files)
                                    progress = 0.5 + (processed / total_files) * 0.5
                                    progress_bar.progress(progress)
                                    status_text.text(f"📊 Processed {processed}/{total_files} files...")
                                
                                # Finalize catalog
                                dataset_manager.catalog.finalize_dataset_catalog(config)
                                
                                progress_bar.progress(1.0)
                                status_text.text("✅ Dataset setup complete!")
                                
                                st.success(f"✅ Successfully setup {dataset_name} with {total_files} audio files")
                                time.sleep(1)  # Brief pause to show completion
                                st.rerun()
                            else:
                                progress_bar.progress(0.0)
                                status_text.text("❌ Download failed")
                                st.error(f"❌ Failed to setup {dataset_name}")
                    
                    except Exception as e:
                        st.error(f"❌ Setup error: {e}")
                        import traceback
                        st.code(traceback.format_exc())
                        
                except Exception as e:
                    st.error(f"❌ Configuration error: {e}")
    
    with tab2:
        st.subheader("Dataset Summary")
        
        if st.button("🔄 Refresh Summary"):
            st.rerun()
        
        try:
            summary = dataset_manager.get_dataset_summary()
            
            # Overview metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Datasets", summary["total_datasets"])
            with col2:
                st.metric("Total Files", summary["total_files"])
            with col3:
                total_duration = sum(
                    file_info.get("duration", 0) 
                    for file_info in dataset_manager.catalog.catalog["files"].values()
                )
                st.metric("Total Duration", f"{total_duration/3600:.1f} hours")
            
            # Datasets table
            if summary["datasets"]:
                st.subheader("Datasets")
                df_datasets = pd.DataFrame.from_dict(summary["datasets"], orient="index")
                st.dataframe(df_datasets, width='stretch')
            
            # Source type distribution
            col1, col2 = st.columns(2)
            
            with col1:
                if summary["source_type_counts"]:
                    st.subheader("Source Types")
                    fig_sources = px.pie(
                        values=list(summary["source_type_counts"].values()),
                        names=list(summary["source_type_counts"].keys()),
                        title="Distribution by Source Type"
                    )
                    st.plotly_chart(fig_sources, width='stretch')
            
            with col2:
                if summary["format_counts"]:
                    st.subheader("Audio Formats")
                    fig_formats = px.bar(
                        x=list(summary["format_counts"].keys()),
                        y=list(summary["format_counts"].values()),
                        title="Files by Format"
                    )
                    st.plotly_chart(fig_formats, width='stretch')
            
        except Exception as e:
            st.error(f"Error getting dataset summary: {e}")
    
    with tab3:
        st.subheader("Browse Audio Files")
        
        # Initialize session state for search results
        if 'browse_search_results' not in st.session_state:
            st.session_state.browse_search_results = None
        if 'browse_selected_file_idx' not in st.session_state:
            st.session_state.browse_selected_file_idx = None
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            filter_source_type = st.selectbox("Filter by Source Type", 
                ["All"] + [t.value for t in SourceType])
        
        with col2:
            min_duration = st.number_input("Min Duration (s)", 0.0, 300.0, 0.0)
            max_duration = st.number_input("Max Duration (s)", 0.1, 300.0, 300.0)
        
        with col3:
            filter_dataset = st.selectbox("Filter by Dataset", 
                ["All"] + list(dataset_manager.catalog.catalog["datasets"].keys()))
        
        # Search button
        if st.button("🔍 Search Files"):
            with st.spinner("Searching files..."):
                try:
                    # Apply filters
                    source_filter = None if filter_source_type == "All" else SourceType(filter_source_type)
                    dataset_filter = None if filter_dataset == "All" else filter_dataset
                    
                    matches = dataset_manager.catalog.find_audio_files(
                        source_type=source_filter,
                        min_duration=min_duration,
                        max_duration=max_duration,
                        dataset_name=dataset_filter
                    )
                    
                    # Store results in session state
                    st.session_state.browse_search_results = matches
                    st.session_state.browse_selected_file_idx = None  # Reset selection
                    
                    # Force rerun to update the display without changing tabs
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error searching files: {e}")
                    st.session_state.browse_search_results = None
        
        # Display search results if available
        if st.session_state.browse_search_results is not None:
            matches = st.session_state.browse_search_results
            
            if matches:
                st.success(f"Found {len(matches)} matching files")
                
                # Convert to DataFrame for display
                df_files = pd.DataFrame(matches)
                
                # Select columns to display
                display_cols = ["absolute_path", "source_type", "duration", 
                               "sample_rate", "format", "dataset"]
                available_cols = [col for col in display_cols if col in df_files.columns]
                
                # Create a cleaner display dataframe
                display_df = df_files[available_cols].copy()
                display_df['filename'] = display_df['absolute_path'].apply(lambda x: Path(x).name)
                
                # Reorder columns to show filename first
                cols_reordered = ['filename'] + [col for col in available_cols if col != 'absolute_path']
                display_df = display_df[cols_reordered]
                
                # Display the metadata table first
                st.write("📋 **Search Results:**")
                st.dataframe(display_df, width='stretch')
                
                # File selection with scrollable container
                st.write("🎵 **Select a file to preview:**")
                st.caption(f"📊 Total results: {len(matches)} files")
                
                # Create a scrollable container using st.container with height
                container = st.container(height=400)
                with container:
                    # Use radio buttons with dataset name included - show all results in scrollable container
                    selected_file_idx = st.radio(
                        "Choose file:",
                        range(len(matches)),
                        format_func=lambda x: f"🎵 {Path(matches[x]['absolute_path']).name} ({matches[x].get('duration', 0):.1f}s) - {matches[x].get('dataset', 'Unknown Dataset')}",
                        key="browse_file_selector",
                        horizontal=False
                    )
                
                # Store selection
                st.session_state.browse_selected_file_idx = selected_file_idx
                
                # Audio player for selected file
                if selected_file_idx is not None and selected_file_idx < len(matches):
                    selected_file = matches[selected_file_idx]["absolute_path"]
                    selected_filename = Path(selected_file).name
                    
                    st.success(f"🎵 **Now Playing:** {selected_filename}")
                    
                    if Path(selected_file).exists():
                        try:
                            with open(selected_file, "rb") as audio_file:
                                audio_bytes = audio_file.read()
                                
                            # Determine audio format from file extension
                            file_ext = Path(selected_file).suffix.lower()
                            if file_ext == '.wav':
                                audio_format = 'audio/wav'
                            elif file_ext == '.mp3':
                                audio_format = 'audio/mp3'
                            elif file_ext == '.ogg':
                                audio_format = 'audio/ogg'
                            else:
                                audio_format = 'audio/wav'  # Default
                            
                            st.audio(audio_bytes, format=audio_format)
                            
                            # Show file info
                            file_info = matches[selected_file_idx]
                            with st.expander("📋 File Details", expanded=True):
                                info_col1, info_col2 = st.columns(2)
                                with info_col1:
                                    st.write(f"**Duration:** {file_info.get('duration', 'Unknown'):.1f} seconds")
                                    st.write(f"**Sample Rate:** {file_info.get('sample_rate', 'Unknown')} Hz")
                                    st.write(f"**File Path:** {file_info.get('absolute_path', 'Unknown')}")
                                with info_col2:
                                    st.write(f"**Format:** {file_info.get('format', 'Unknown')}")
                                    st.write(f"**Dataset:** {file_info.get('dataset', 'Unknown')}")
                                    st.write(f"**Source Type:** {file_info.get('source_type', 'Unknown')}")
                                    
                        except Exception as e:
                            st.error(f"Error loading audio file: {e}")
                    else:
                        st.error(f"File not found: {selected_file}")
            else:
                st.warning("No files match the specified criteria")
        else:
            st.info("👆 Use the filters above and click 'Search Files' to find audio files")
    
    with tab4:
        st.subheader("Dataset Settings")
        
        cache_dir = st.text_input("Cache Directory", 
                                 value=str(dataset_manager.cache_dir))
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🧹 Clear Cache"):
                if st.checkbox("I understand this will delete all cached data"):
                    # Implementation would go here
                    st.warning("Cache clearing not implemented in demo")
        
        with col2:
            if st.button("📊 Rebuild Catalog"):
                with st.spinner("Rebuilding catalog..."):
                    # Implementation would go here
                    st.info("Catalog rebuild not implemented in demo")

def scene_creator_page():
    """Scene creator page for building acoustic scenes"""
    
    st.markdown('<h2 class="section-header">🎬 Scene Creator</h2>', unsafe_allow_html=True)
    
    # Initialize dataset manager
    if st.session_state.dataset_manager is None:
        with st.spinner("Initializing dataset manager..."):
            st.session_state.dataset_manager = DatasetManager()
    
    dataset_manager = st.session_state.dataset_manager
    
    # Render the scene creator interface
    render_scene_creator_tab(dataset_manager)

def configuration_page():
    """Configuration page for creating simulation scenarios"""
    
    st.markdown('<h2 class="section-header">⚙️ Configuration</h2>', unsafe_allow_html=True)
    
    # Configuration tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Basic Setup", "🌍 Environment", "🔊 Sources", "📝 Review"])
    
    # Initialize session state for config
    if 'config_draft' not in st.session_state:
        st.session_state.config_draft = {
            'basic': {},
            'environment': {},
            'sources': [],
            'ambient': {}
        }
    
    with tab1:
        st.subheader("Basic Scene Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            scene_name = st.text_input("Scene Name", "my_acoustic_scene")
            scene_duration = st.slider("Duration (seconds)", 1.0, 60.0, 10.0, 0.5)
            
            # Processing parameters
            st.subheader("Processing Parameters")
            sample_rate = st.selectbox("Sample Rate", [16000, 22050, 44100, 48000])
            frame_size = st.selectbox("Frame Size", [256, 512, 1024, 2048])
            
        with col2:
            description = st.text_area("Description", 
                "Describe your acoustic scene scenario...")
            
            background_noise = st.slider("Background Noise (dB)", 0.0, 80.0, 45.0, 1.0)
            
            # Tags
            tags_input = st.text_input("Tags (comma-separated)", 
                "simulation, wildlife, validation")
            tags = [tag.strip() for tag in tags_input.split(",") if tag.strip()]
        
        # Store basic config
        st.session_state.config_draft['basic'] = {
            'scene_name': scene_name,
            'scene_duration': scene_duration,
            'description': description,
            'background_noise_level': background_noise,
            'tags': tags,
            'sample_rate': sample_rate,
            'frame_size': frame_size
        }
    
    with tab2:
        st.subheader("Acoustic Environment")
        
        # Environment presets
        env_preset = st.selectbox("Environment Preset", [
            "Custom", "Forest Clearing", "Dense Forest", "Savanna"
        ])
        
        col1, col2 = st.columns(2)
        
        with col1:
            if env_preset == "Custom":
                room_x = st.number_input("Room Width (m)", 100.0, 10000.0, 1000.0)
                room_y = st.number_input("Room Length (m)", 100.0, 10000.0, 1000.0)
                room_z = st.number_input("Room Height (m)", 10.0, 500.0, 50.0)
                absorption = st.slider("Absorption Coefficient", 0.001, 0.5, 0.02, 0.001)
                max_order = st.slider("Max Reflection Order", 0, 10, 2)
            else:
                # Use preset values but allow viewing
                presets = {
                    "Forest Clearing": (1000, 1000, 50, 0.02, 2),
                    "Dense Forest": (800, 800, 30, 0.08, 3),
                    "Savanna": (2000, 2000, 100, 0.01, 1)
                }
                room_x, room_y, room_z, absorption, max_order = presets[env_preset]
                
                st.info(f"Using {env_preset} preset:")
                st.write(f"- Size: {room_x}×{room_y}×{room_z}m")
                st.write(f"- Absorption: {absorption}")
                st.write(f"- Max reflections: {max_order}")
        
        with col2:
            temperature = st.slider("Temperature (°C)", -10.0, 50.0, 25.0, 1.0)
            humidity = st.slider("Humidity (%)", 0.0, 100.0, 60.0, 5.0)
            air_absorption = st.checkbox("Air Absorption", True)
            
            # Ambient conditions
            st.subheader("Ambient Conditions")
            
            add_rain = st.checkbox("Add Rain")
            if add_rain:
                rain_intensity = st.selectbox("Rain Intensity", 
                    ["light", "moderate", "heavy"])
                rain_level = st.slider("Rain Level (dB)", 10.0, 60.0, 35.0)
            
            add_wind = st.checkbox("Add Wind")
            if add_wind:
                wind_speed = st.slider("Wind Speed (km/h)", 0.0, 50.0, 15.0)
                wind_level = st.slider("Wind Level (dB)", 10.0, 50.0, 28.0)
        
        # Store environment config
        ambient_config = {}
        if add_rain:
            ambient_config['rain'] = {
                'condition_type': 'rain',
                'intensity': rain_intensity,
                'level_db': rain_level
            }
        if add_wind:
            ambient_config['wind'] = {
                'condition_type': 'wind',
                'intensity': wind_speed,
                'level_db': wind_level
            }
        
        st.session_state.config_draft['environment'] = {
            'environment_type': env_preset.lower().replace(' ', '_'),
            'room_size': [room_x, room_y, room_z],
            'absorption_coefficient': absorption,
            'max_order': max_order,
            'temperature': temperature,
            'humidity': humidity,
            'air_absorption': air_absorption
        }
        st.session_state.config_draft['ambient'] = ambient_config
    
    with tab3:
        st.subheader("Sound Sources")
        
        # Source management
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.write("**Current Sources:**")
            
            # Display current sources
            if st.session_state.config_draft['sources']:
                for i, source in enumerate(st.session_state.config_draft['sources']):
                    with st.expander(f"Source {i+1}: {source.get('source_type', 'unknown')}"):
                        st.write(f"**Audio:** {source.get('audio_file', 'N/A')}")
                        st.write(f"**Position:** ({source.get('position', [0,0,0])[0]:.0f}, {source.get('position', [0,0,0])[1]:.0f}, {source.get('position', [0,0,0])[2]:.0f}) m")
                        st.write(f"**Start Time:** {source.get('start_time', 0):.1f}s")
                        st.write(f"**Volume:** {source.get('volume_scale', 1.0):.1f}")
                        
                        if st.button(f"❌ Remove", key=f"remove_{i}"):
                            st.session_state.config_draft['sources'].pop(i)
                            st.rerun()
            else:
                st.info("No sources added yet. Add sources using the form below.")
        
        with col2:
            if st.button("🗑️ Clear All Sources"):
                st.session_state.config_draft['sources'] = []
                st.rerun()
        
        # Add new source form
        st.subheader("Add New Source")
        
        with st.form("add_source_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                source_type = st.selectbox("Source Type", [t.value for t in SourceType])
                
                # Smart audio file selection
                if st.session_state.dataset_manager:
                    use_smart_selection = st.checkbox("Use Smart Audio Selection", True)
                    
                    if use_smart_selection:
                        audio_file = "auto_select"
                        st.info("Audio file will be automatically selected based on source type")
                    else:
                        audio_file = st.text_input("Audio File Path", 
                            placeholder="sounds/elephant_call.wav")
                else:
                    audio_file = st.text_input("Audio File Path", 
                        placeholder="sounds/elephant_call.wav")
                
                # Position using polar coordinates (easier for users)
                distance = st.number_input("Distance (m)", 10.0, 1000.0, 100.0)
                azimuth = st.slider("Azimuth (degrees)", 0, 360, 45)
                
            with col2:
                elevation = st.number_input("Elevation (m)", -10.0, 100.0, 0.0)
                start_time = st.number_input("Start Time (s)", 0.0, 60.0, 2.0)
                volume_scale = st.slider("Volume Scale", 0.1, 3.0, 1.0, 0.1)
                duration_override = st.number_input("Duration Override (s, 0=use original)", 
                                                  0.0, 60.0, 0.0)
            
            if st.form_submit_button("➕ Add Source"):
                # Convert polar to cartesian
                azimuth_rad = np.radians(azimuth)
                x = distance * np.cos(azimuth_rad)
                y = distance * np.sin(azimuth_rad)
                
                new_source = {
                    'audio_file': audio_file,
                    'position': [x, y, elevation],
                    'start_time': start_time,
                    'volume_scale': volume_scale,
                    'source_type': source_type,
                    'duration_override': duration_override if duration_override > 0 else None
                }
                
                st.session_state.config_draft['sources'].append(new_source)
                st.success(f"✅ Added {source_type} source")
                st.rerun()
    
    with tab4:
        st.subheader("Configuration Review")
        
        # Build complete configuration
        try:
            # Prepare sources
            sources = []
            for source_data in st.session_state.config_draft['sources']:
                if source_data['audio_file'] == "auto_select" and st.session_state.dataset_manager:
                    # Smart audio selection
                    selector = SmartAudioSelector(st.session_state.dataset_manager)
                    
                    if source_data['source_type'] == 'elephant':
                        selected_file = selector.select_elephant_audio("general")
                    else:
                        source_type = SourceType(source_data['source_type'])
                        selected_file = st.session_state.dataset_manager.catalog.get_random_audio_file(
                            source_type, min_duration=1.0
                        )
                    
                    if selected_file:
                        source_data = source_data.copy()
                        source_data['audio_file'] = selected_file
                    else:
                        st.warning(f"Could not find audio for {source_data['source_type']}")
                        continue
                
                sources.append(SoundSource(**source_data))
            
            # Build environment
            env_config = EnvironmentConfig(**st.session_state.config_draft['environment'])
            
            # Build ambient conditions
            ambient = {}
            for name, amb_data in st.session_state.config_draft['ambient'].items():
                ambient[name] = AmbientCondition(**amb_data)
            
            # Build complete configuration
            config = SimulationConfig(
                scene_name=st.session_state.config_draft['basic']['scene_name'],
                scene_duration=st.session_state.config_draft['basic']['scene_duration'],
                background_noise_level=st.session_state.config_draft['basic']['background_noise_level'],
                sources=sources,
                environment=env_config,
                ambient=ambient,
                description=st.session_state.config_draft['basic'].get('description'),
                tags=st.session_state.config_draft['basic'].get('tags', []),
                processing=ProcessingConfig(
                    sample_rate=st.session_state.config_draft['basic']['sample_rate'],
                    frame_size=st.session_state.config_draft['basic']['frame_size']
                )
            )
            
            st.success("✅ Configuration is valid!")
            
            # Display summary
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Sources", len(config.sources))
                st.metric("Duration", f"{config.scene_duration}s")
            
            with col2:
                st.metric("Environment", config.environment.environment_type.value.title())
                st.metric("Sample Rate", f"{config.processing.sample_rate} Hz")
            
            with col3:
                st.metric("Ambient Conditions", len(config.ambient))
                st.metric("Background Noise", f"{config.background_noise_level} dB")
            
            # Configuration JSON
            with st.expander("📄 View Configuration JSON"):
                config_dict = config.model_dump()
                st.json(config_dict)
            
            # Action buttons
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("💾 Save Configuration"):
                    # Store in session state for simulation
                    st.session_state.current_config = config
                    st.success("Configuration saved! Go to Simulation tab to run.")
            
            with col2:
                if st.button("📁 Export Configuration"):
                    config_json = json.dumps(config.model_dump(), indent=2, default=str)
                    st.download_button(
                        "Download Config JSON",
                        config_json,
                        file_name=f"{config.scene_name}_config.json",
                        mime="application/json"
                    )
            
            with col3:
                uploaded_config = st.file_uploader("📤 Import Configuration", 
                                                  type=["json"], key="config_import")
                
                if uploaded_config is not None:
                    try:
                        imported_config = json.load(uploaded_config)
                        config = SimulationConfig(**imported_config)
                        st.session_state.current_config = config
                        st.success("Configuration imported successfully!")
                    except Exception as e:
                        st.error(f"Error importing configuration: {e}")
        
        except Exception as e:
            st.error(f"❌ Configuration validation failed: {e}")
            st.info("Please check your configuration in the previous tabs.")

def analysis_page():
    """Analysis and visualization page"""
    
    st.markdown('<h2 class="section-header">📈 Analysis & Visualization</h2>', unsafe_allow_html=True)
    
    if not st.session_state.simulation_results:
        st.warning("⚠️ No simulation results available. Please run a simulation first.")
        return
    
    # Select simulation to analyze
    available_results = list(st.session_state.simulation_results.keys())
    selected_result = st.selectbox("Select Simulation Results", available_results)
    
    if selected_result not in st.session_state.simulation_results:
        return
    
    result_data = st.session_state.simulation_results[selected_result]
    mic_signals = result_data['mic_signals']
    metadata = result_data['metadata']
    config = result_data['config']
    validation = result_data.get('validation', {})
    
    # Analysis tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Signal Analysis", "🎯 DOA Analysis", "🗺️ Scene Visualization", "📋 Detailed Results"])
    
    with tab1:
        st.subheader("Signal Analysis")
        
        # Time domain plot
        fig_time = make_subplots(
            rows=4, cols=1,
            subplot_titles=[f'Microphone {i+1}' for i in range(4)],
            vertical_spacing=0.05
        )
        
        time_axis = np.arange(mic_signals.shape[1]) / config.processing.sample_rate
        
        for i in range(4):
            fig_time.add_trace(
                go.Scatter(x=time_axis, y=mic_signals[i], 
                          name=f'Mic {i+1}', line=dict(width=1)),
                row=i+1, col=1
            )
        
        fig_time.update_layout(height=600, title="Microphone Array Signals")
        fig_time.update_xaxes(title="Time (s)", row=4, col=1)
        fig_time.update_yaxes(title="Amplitude")
        
        st.plotly_chart(fig_time, width='stretch')
        
        # Frequency analysis
        st.subheader("Frequency Analysis")
        
        # Compute PSD for first microphone
        from scipy import signal as scipy_signal
        
        freqs, psd = scipy_signal.welch(
            mic_signals[0], config.processing.sample_rate, 
            nperseg=1024, noverlap=512
        )
        
        fig_freq = go.Figure()
        fig_freq.add_trace(go.Scatter(
            x=freqs, y=10*np.log10(psd + 1e-12),
            name='Mic 1 PSD'
        ))
        
        fig_freq.update_layout(
            title="Power Spectral Density",
            xaxis_title="Frequency (Hz)",
            yaxis_title="Power (dB)",
            xaxis_type="log"
        )
        
        st.plotly_chart(fig_freq, width='stretch')
    
    with tab2:
        if validation:
            st.subheader("Direction of Arrival Analysis")
            
            # Ground truth vs estimates comparison
            ground_truth = metadata['ground_truth']['source_directions']
            
            if ground_truth:
                fig_doa = go.Figure()
                
                # Ground truth
                gt_azimuths = [src['azimuth_deg'] for src in ground_truth]
                fig_doa.add_trace(go.Scatter(
                    x=list(range(len(gt_azimuths))),
                    y=gt_azimuths,
                    mode='markers+lines',
                    name='Ground Truth',
                    marker=dict(size=12, color='black', symbol='star')
                ))
                
                # Algorithm estimates
                colors = ['red', 'blue', 'green', 'orange']
                for i, (method, results) in enumerate(validation.items()):
                    if 'azimuth_estimates' in results:
                        estimates = results['azimuth_estimates']
                        if estimates:
                            fig_doa.add_trace(go.Scatter(
                                x=list(range(len(estimates))),
                                y=estimates,
                                mode='markers',
                                name=method,
                                marker=dict(size=10, color=colors[i % len(colors)])
                            ))
                
                fig_doa.update_layout(
                    title="DOA Estimation Comparison",
                    xaxis_title="Source Index",
                    yaxis_title="Azimuth (degrees)",
                    yaxis_range=[0, 360]
                )
                
                st.plotly_chart(fig_doa, width='stretch')
                
                # Error analysis
                st.subheader("Estimation Errors")
                
                error_data = []
                for method, results in validation.items():
                    if 'azimuth_estimates' in results:
                        estimates = results['azimuth_estimates']
                        for i, (est, gt) in enumerate(zip(estimates, gt_azimuths)):
                            error = abs(est - gt)
                            if error > 180:
                                error = 360 - error
                            
                            error_data.append({
                                'Method': method,
                                'Source': i+1,
                                'Error': error
                            })
                
                if error_data:
                    df_errors = pd.DataFrame(error_data)
                    
                    fig_errors = px.bar(
                        df_errors, x='Source', y='Error', color='Method',
                        title="DOA Estimation Errors by Method",
                        labels={'Error': 'Error (degrees)'}
                    )
                    
                    st.plotly_chart(fig_errors, width='stretch')
        
        else:
            st.info("No DOA validation results available")
    
    with tab3:
        st.subheader("Scene Layout Visualization")
        
        # 3D scene plot
        fig_3d = go.Figure()
        
        # Microphone array
        mic_positions = metadata['mic_positions']
        mic_x = [mic['x'] for mic in mic_positions]
        mic_y = [mic['y'] for mic in mic_positions]
        mic_z = [mic['z'] for mic in mic_positions]
        
        fig_3d.add_trace(go.Scatter3d(
            x=mic_x, y=mic_y, z=mic_z,
            mode='markers',
            name='Microphones',
            marker=dict(size=8, color='red', symbol='square')
        ))
        
        # Sources
        ground_truth = metadata['ground_truth']['source_directions']
        if ground_truth:
            for i, src in enumerate(ground_truth):
                pos = src['position']
                # Convert from room coordinates to relative coordinates
                room_center = np.array(config.environment.room_size) / 2
                rel_pos = np.array(pos) - room_center
                
                fig_3d.add_trace(go.Scatter3d(
                    x=[rel_pos[0]], y=[rel_pos[1]], z=[rel_pos[2]],
                    mode='markers+text',
                    name=f'Source {i+1}',
                    text=[f'S{i+1}'],
                    marker=dict(size=6, color=f'rgb({50+i*40}, {100+i*30}, {150+i*20})')
                ))
        
        fig_3d.update_layout(
            title="3D Scene Layout",
            scene=dict(
                xaxis_title="X (m)",
                yaxis_title="Y (m)", 
                zaxis_title="Z (m)"
            )
        )
        
        st.plotly_chart(fig_3d, width='stretch')
        
        # 2D top-down view
        fig_2d = go.Figure()
        
        # Microphones
        fig_2d.add_trace(go.Scatter(
            x=mic_x, y=mic_y,
            mode='markers',
            name='Microphones',
            marker=dict(size=12, color='red', symbol='square')
        ))
        
        # Sources
        if ground_truth:
            for i, src in enumerate(ground_truth):
                pos = src['position']
                room_center = np.array(config.environment.room_size) / 2
                rel_pos = np.array(pos) - room_center
                
                fig_2d.add_trace(go.Scatter(
                    x=[rel_pos[0]], y=[rel_pos[1]],
                    mode='markers+text',
                    name=f'Source {i+1}',
                    text=[f'S{i+1}<br>{src["azimuth_deg"]:.0f}°'],
                    marker=dict(size=10)
                ))
        
        # Add range circles
        for radius in [50, 100, 200, 500]:
            theta = np.linspace(0, 2*np.pi, 100)
            x_circle = radius * np.cos(theta)
            y_circle = radius * np.sin(theta)
            
            fig_2d.add_trace(go.Scatter(
                x=x_circle, y=y_circle,
                mode='lines',
                name=f'{radius}m',
                line=dict(dash='dash', color='gray', width=1),
                showlegend=(radius == 100)  # Only show one in legend
            ))
        
        fig_2d.update_layout(
            title="Top-Down Scene View",
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            xaxis=dict(scaleanchor="y", scaleratio=1)
        )
        
        st.plotly_chart(fig_2d, width='stretch')
    
    with tab4:
        st.subheader("Detailed Results")
        
        # Configuration details
        with st.expander("Configuration Used"):
            st.json(config.model_dump())
        
        # Full metadata
        with st.expander("Simulation Metadata"):
            # Remove large arrays for display
            display_metadata = metadata.copy()
            if 'mic_signals' in display_metadata:
                del display_metadata['mic_signals']
            st.json(display_metadata)
        
        # Download results
        st.subheader("Download Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Download audio files
            if st.button("📁 Download Audio Files"):
                # Create ZIP with all audio channels
                zip_buffer = io.BytesIO()
                
                with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                    for i in range(mic_signals.shape[0]):
                        # Convert to WAV bytes
                        import soundfile as sf
                        audio_buffer = io.BytesIO()
                        sf.write(audio_buffer, mic_signals[i], 
                                config.processing.sample_rate, format='WAV')
                        
                        zip_file.writestr(f"mic_{i+1}.wav", audio_buffer.getvalue())
                
                st.download_button(
                    "Download Audio ZIP",
                    zip_buffer.getvalue(),
                    file_name=f"{selected_result}_audio.zip",
                    mime="application/zip"
                )
        
        with col2:
            # Download metadata
            metadata_json = json.dumps(metadata, indent=2, default=str)
            st.download_button(
                "📄 Download Metadata JSON",
                metadata_json,
                file_name=f"{selected_result}_metadata.json",
                mime="application/json"
            )
        
        with col3:
            # Download NumPy array
            array_buffer = io.BytesIO()
            np.save(array_buffer, mic_signals)
            
            st.download_button(
                "🔢 Download NumPy Array",
                array_buffer.getvalue(),
                file_name=f"{selected_result}_signals.npy",
                mime="application/octet-stream"
            )

def export_import_page():
    """Export and import page"""
    
    st.markdown('<h2 class="section-header">💾 Export & Import</h2>', unsafe_allow_html=True)
    
    # Export section
    st.subheader("Export Results")
    
    if st.session_state.simulation_results:
        export_selection = st.multiselect(
            "Select results to export",
            list(st.session_state.simulation_results.keys())
        )
        
        if export_selection:
            export_format = st.selectbox("Export Format", 
                ["Complete Package (ZIP)", "Audio Only", "Metadata Only", "Configuration Only"],
                index=None)
            
            if st.button("🎁 Create Export Package"):
                # Implementation would create export package
                st.success("Export package created! (Implementation needed)")
    
    else:
        st.info("No simulation results available to export")
    
    # Import section
    st.subheader("Import Data")
    
    uploaded_file = st.file_uploader("Upload simulation package", 
                                    type=["zip", "json"])
    
    if uploaded_file is not None:
        # Implementation would handle import
        st.info("Import functionality not implemented in demo")

def help_page():
    """Help and documentation page"""
    
    st.markdown('<h2 class="section-header">📚 Help & Documentation</h2>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["🚀 Quick Start", "📖 User Guide", "🔧 Technical", "❓ FAQ"])
    
    with tab1:
        st.markdown("""
## Quick Start Guide

### 1. Setup Datasets (Optional but Recommended)
- Go to **Dataset Manager** tab
- Add audio datasets from Kaggle, Google Drive, or local files
- Let the system catalog your audio files

### 2. Create Configuration
- Go to **Configuration** tab
- Configure scene parameters, environment, and sound sources
- Save or export your configuration when ready

### 3. Run Simulation
- Go to **Simulation** tab and click "Run Simulation"
- Optionally enable DOA validation and visualization plots

### 4. Analyze & Export
- Inspect results in **Analysis** tab (signals, DOA, scene layout)
- Export audio, metadata, or numpy arrays via **Export/Import**

For more details, check the User Guide and Technical tabs.
""", unsafe_allow_html=True)

class SimulationSession:
    """Manages simulation session state"""
    
    def __init__(self):
        if 'simulation_results' not in st.session_state:
            st.session_state.simulation_results = {}
        if 'dataset_manager' not in st.session_state:
            st.session_state.dataset_manager = None
        if 'current_config' not in st.session_state:
            st.session_state.current_config = None

def initialize_app():
    """Initialize the application"""
    
    st.markdown('<h1 class="main-header">🐘 Wildlife Acoustic Scene Simulator</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    **Create realistic acoustic scenes for wildlife monitoring algorithm validation**
    
    This tool generates synthetic microphone array data with:
    - Realistic 3D acoustic propagation using pyroomacoustics
    - Real wildlife audio from datasets (Kaggle, Google Drive, local files)
    - Ground truth DOA information for algorithm validation
    - Built-in DOA algorithm comparison (MUSIC, SRP-PHAT, FRIDA)
    """)

def sidebar_navigation():
    """Create sidebar navigation"""
    
    st.sidebar.title("🎛️ Navigation")
    
    pages = {
        "🏠 Home": "home",
        "📊 Dataset Manager": "datasets",
        "🎬 Scene Creator": "scene_creator",
        "🤖 Scene Generator": "scene_generator",
        "📈 Analysis": "analysis",
        "💾 Export/Import": "export",
        "📚 Help": "help"
    }
    
    selected = st.sidebar.selectbox("Select Page", list(pages.keys()))
    return pages[selected]

def configuration_page():
    """Configuration page for creating simulation scenarios"""
    
    st.markdown('<h2 class="section-header">⚙️ Configuration</h2>', unsafe_allow_html=True)
    
    # Configuration tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Basic Setup", "🌍 Environment", "🔊 Sources", "📝 Review"])
    
    # Initialize session state for config
    if 'config_draft' not in st.session_state:
        st.session_state.config_draft = {
            'basic': {},
            'environment': {},
            'sources': [],
            'ambient': {}
        }
    
    with tab1:
        st.subheader("Basic Scene Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            scene_name = st.text_input("Scene Name", "my_acoustic_scene")
            scene_duration = st.slider("Duration (seconds)", 1.0, 60.0, 10.0, 0.5)
            
            # Processing parameters
            st.subheader("Processing Parameters")
            sample_rate = st.selectbox("Sample Rate", [16000, 22050, 44100, 48000])
            frame_size = st.selectbox("Frame Size", [256, 512, 1024, 2048])
            
        with col2:
            description = st.text_area("Description", 
                "Describe your acoustic scene scenario...")
            
            background_noise = st.slider("Background Noise (dB)", 0.0, 80.0, 45.0, 1.0)
            
            # Tags
            tags_input = st.text_input("Tags (comma-separated)", 
                "simulation, wildlife, validation")
            tags = [tag.strip() for tag in tags_input.split(",") if tag.strip()]
        
        # Store basic config
        st.session_state.config_draft['basic'] = {
            'scene_name': scene_name,
            'scene_duration': scene_duration,
            'description': description,
            'background_noise_level': background_noise,
            'tags': tags,
            'sample_rate': sample_rate,
            'frame_size': frame_size
        }
    
    with tab2:
        st.subheader("Acoustic Environment")
        
        # Environment presets
        env_preset = st.selectbox("Environment Preset", [
            "Custom", "Forest Clearing", "Dense Forest", "Savanna"
        ])
        
        col1, col2 = st.columns(2)
        
        with col1:
            if env_preset == "Custom":
                room_x = st.number_input("Room Width (m)", 100.0, 10000.0, 1000.0)
                room_y = st.number_input("Room Length (m)", 100.0, 10000.0, 1000.0)
                room_z = st.number_input("Room Height (m)", 10.0, 500.0, 50.0)
                absorption = st.slider("Absorption Coefficient", 0.001, 0.5, 0.02, 0.001)
                max_order = st.slider("Max Reflection Order", 0, 10, 2)
            else:
                # Use preset values but allow viewing
                presets = {
                    "Forest Clearing": (1000, 1000, 50, 0.02, 2),
                    "Dense Forest": (800, 800, 30, 0.08, 3),
                    "Savanna": (2000, 2000, 100, 0.01, 1)
                }
                room_x, room_y, room_z, absorption, max_order = presets[env_preset]
                
                st.info(f"Using {env_preset} preset:")
                st.write(f"- Size: {room_x}×{room_y}×{room_z}m")
                st.write(f"- Absorption: {absorption}")
                st.write(f"- Max reflections: {max_order}")
        
        with col2:
            temperature = st.slider("Temperature (°C)", -10.0, 50.0, 25.0, 1.0)
            humidity = st.slider("Humidity (%)", 0.0, 100.0, 60.0, 5.0)
            air_absorption = st.checkbox("Air Absorption", True)
            
            # Ambient conditions
            st.subheader("Ambient Conditions")
            
            add_rain = st.checkbox("Add Rain")
            if add_rain:
                rain_intensity = st.selectbox("Rain Intensity", 
                    ["light", "moderate", "heavy"])
                rain_level = st.slider("Rain Level (dB)", 10.0, 60.0, 35.0)
            
            add_wind = st.checkbox("Add Wind")
            if add_wind:
                wind_speed = st.slider("Wind Speed (km/h)", 0.0, 50.0, 15.0)
                wind_level = st.slider("Wind Level (dB)", 10.0, 50.0, 28.0)

def main():
    """Main Streamlit application"""
    
    # Initialize session
    session = SimulationSession()
    
    # Initialize app
    initialize_app()
    
    # Navigation
    page = sidebar_navigation()
    
    # Route to appropriate page
    if page == "home":
        st.markdown("### Welcome to the Wildlife Acoustic Scene Simulator!")
        st.markdown("""
        A comprehensive toolkit for generating realistic acoustic scenes for wildlife monitoring research.
        
        #### 🎯 **Available Tools**
        
        **📊 Dataset Manager**: Add and manage audio datasets from Kaggle, Google Drive, or local files.
        
        **🎬 Scene Creator**: Interactively create individual acoustic scenes with precise sound placement and preview.
        
        **🤖 Scene Generator**: Generate large datasets (100s-1000s) of elephant detection scenes automatically with:
        - Configurable positive/negative sample ratios
        - Physics-based sound positioning with height constraints
        - Realistic ambient background sounds
        - 4-microphone array audio output
        - JSONL configuration format for machine learning workflows
        
        **📈 Analysis**: Visualize and analyze simulation results with DOA estimation and signal processing tools.
        
        **💾 Export/Import**: Save and load scene configurations, export audio files and metadata.
        
        #### 🚀 **Quick Start for ML Training Data**
        1. **Dataset Manager** → Add your audio datasets
        2. **Scene Generator** → Generate training scenes with elephant detection labels
        3. **Analysis** → Validate generated scenes and ground truth
        
        Perfect for training machine learning models for wildlife acoustic monitoring!
        """)
    elif page == "datasets":
        dataset_manager_page()
    elif page == "scene_creator":
        scene_creator_page()
    elif page == "scene_generator":
        dataset_manager = DatasetManager()
        render_scene_generator_page(dataset_manager)
    elif page == "analysis":
        analysis_page()
    elif page == "export":
        export_import_page()
    elif page == "help":
        help_page()

if __name__ == "__main__":
    main()