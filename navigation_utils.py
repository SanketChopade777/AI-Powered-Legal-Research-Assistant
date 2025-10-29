import streamlit as st

def run_external_app(app_file):
    """Navigate to external app"""
    st.session_state.current_page = app_file
    st.rerun()

def show_external_app_page(load_and_run_module, show_fallback_interface):
    """Show the external app with a back button"""
    # Back button at top
    # col1, col2, col3 = st.columns([1, 2, 1])
    # with col1:
    #     if st.button("← Back to Home", key="back_btn"):
    #         st.session_state.current_page = 'home'
    #         st.rerun()

    # Run the external app
    app_file = st.session_state.current_page

    # Check if file exists
    import os
    if not os.path.exists(app_file):
        st.error(f"File {app_file} not found! Please make sure it exists in the same directory.")
        show_fallback_interface(app_file)
        return

    try:
        load_and_run_module(app_file)
    except Exception as e:
        st.error(f"Error running {app_file}: {str(e)}")
        show_fallback_interface(app_file)