"""
Integration test for wildlife acoustic simulation components
"""

import sys

def test_enhanced_models():
    """Test Pydantic models"""
    try:
        from models import SimulationConfig, create_elephant_source, create_forest_environment
        
        # Test model creation
        elephant_source = create_elephant_source(100, 45, "test.wav")
        forest_env = create_forest_environment()
        
        config = SimulationConfig(
            scene_duration=10.0,
            sources=[elephant_source],
            environment=forest_env,
            scene_name="test_scene"
        )
        
        print("✅ Enhanced Models: Working")
        return True
        
    except Exception as e:
        print(f"❌ Enhanced Models: {e}")
        return False

def test_dataset_manager():
    """Test dataset manager"""
    try:
        from dataset_manager import DatasetManager
        
        manager = DatasetManager()
        summary = manager.get_dataset_summary()
        
        print("✅ Dataset Manager: Working")
        return True
        
    except Exception as e:
        print(f"❌ Dataset Manager: {e}")
        return False

def test_streamlit_app():
    """Test basic Streamlit app imports"""
    try:
        import streamlit as st
        import plotly.graph_objects as go
        import plotly.express as px
        
        print("✅ Streamlit Dependencies: Working")
        return True
        
    except Exception as e:
        print(f"❌ Streamlit Dependencies: {e}")
        return False

def test_hydra_basic():
    """Test basic Hydra functionality"""
    try:
        import hydra
        from omegaconf import OmegaConf
        
        # Test basic OmegaConf usage
        cfg = OmegaConf.create({"test": "value"})
        
        print("✅ Hydra Basic: Working")
        return True
        
    except Exception as e:
        print(f"❌ Hydra Basic: {e}")
        return False

def run_integration_tests():
    """Run all integration tests"""
    
    print("Running Integration Tests...")
    print("=" * 50)
    
    tests = [
        test_enhanced_models,
        test_dataset_manager,
        test_streamlit_app,
        test_hydra_basic
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("=" * 50)
    passed = sum(results)
    total = len(results)
    
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All integration tests passed!")
        return True
    else:
        print("⚠️ Some integration tests failed")
        return False

if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)