"""
Simple Streamlit App for Testing Wildlife Acoustic Simulation
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import json
from datetime import datetime

# Import our modules
try:
    from models import *
    from dataset_manager import DatasetManager
    from acoustic_scene_generator import WildlifeAcousticSimulator
    MODULES_AVAILABLE = True
except ImportError as e:
    st.error(f"Could not import modules: {e}")
    MODULES_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Wildlife Acoustic Simulation",
    page_icon="🐘",
    layout="wide"
)

def main():
    """Simple main application for testing"""
    
    st.title("🐘 Wildlife Acoustic Scene Simulator")
    st.markdown("**Test Interface for Research Components**")
    
    if not MODULES_AVAILABLE:
        st.error("Required modules not available. Please check imports.")
        return
    
    # Test sections
    tab1, tab2, tab3 = st.tabs(["🧪 Model Testing", "📊 Dataset Manager", "🎬 Basic Simulation"])
    
    with tab1:
        st.subheader("Pydantic Model Testing")
        
        if st.button("Test SimulationConfig Creation"):
            try:
                # Create test configuration
                elephant_source = create_elephant_source(100, 45, "test_elephant.wav")
                forest_env = create_forest_environment()
                
                config = SimulationConfig(
                    scene_duration=10.0,
                    sources=[elephant_source],
                    environment=forest_env,
                    scene_name="test_scene"
                )
                
                st.success("✅ SimulationConfig created successfully!")
                st.json(config.dict())
                
            except Exception as e:
                st.error(f"❌ Model creation failed: {e}")
    
    with tab2:
        st.subheader("Dataset Manager Testing")
        
        if st.button("Initialize Dataset Manager"):
            try:
                dataset_manager = DatasetManager()
                st.success("✅ Dataset Manager initialized!")
                st.info(f"Cache directory: {dataset_manager.cache_dir}")
                
            except Exception as e:
                st.error(f"❌ Dataset Manager initialization failed: {e}")
    
    with tab3:
        st.subheader("Basic Simulation Testing")
        
        if st.button("Test Simulator Import"):
            try:
                simulator = WildlifeAcousticSimulator()
                st.success("✅ WildlifeAcousticSimulator imported successfully!")
                st.info("Simulator ready for use")
                
            except Exception as e:
                st.error(f"❌ Simulator initialization failed: {e}")
    
    # Component status
    st.sidebar.header("Component Status")
    
    components = [
        ("Enhanced Models", "enhanced_models"),
        ("Dataset Manager", "dataset_manager"), 
        ("Acoustic Generator", "acoustic_scene_generator")
    ]
    
    for name, module_name in components:
        try:
            __import__(module_name)
            st.sidebar.success(f"✅ {name}")
        except ImportError:
            st.sidebar.error(f"❌ {name}")

if __name__ == "__main__":
    main()