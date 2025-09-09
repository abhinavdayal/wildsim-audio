def analysis_page():
    """Analysis and visualization page"""
    
    st.markdown('<h2 class="section-header">🎯 ZODAS Direction of Arrival Analysis</h2>', unsafe_allow_html=True)
    
    # Import and run ZODAS UI
    from zodas_ui import run_zodas_dataset_analysis_ui, zodas_sidebar_help
    zodas_sidebar_help()
    run_zodas_dataset_analysis_ui()
