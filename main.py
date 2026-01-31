import sys
import streamlit as st
import importlib.util
import os

# Import from our modules
from styles import inject_navigation_css
from home_page import show_home_page
from navigation_utils import show_external_app_page
from fallback_components import show_chatbot_fallback, show_document_fallback, show_lawyer_fallback


def load_and_run_module(module_path):
    """Dynamically load and run a Python file"""
    try:
        module_name = os.path.basename(module_path).replace('.py', '')

        # Clear module cache if it exists
        if module_name in sys.modules:
            del sys.modules[module_name]

        # Load and execute the module
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None:
            raise ImportError(f"Could not load spec from {module_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        # Check if main function exists
        if hasattr(module, 'main'):
            module.main()
        else:
            st.error(f"No 'main' function found in {module_path}")

    except Exception as e:
        st.error(f"Error loading {module_path}: {str(e)}")
        # Show fallback interface
        show_fallback_interface(module_path)


def show_fallback_interface(module_name):
    """Show a fallback interface when module fails to load"""
    if 'document' in module_name:
        show_document_fallback()
    elif 'chatbot' in module_name or 'main' in module_name:
        show_chatbot_fallback()
    elif 'lawyer' in module_name:
        show_lawyer_fallback()
    else:
        st.info(f"Module {module_name} would run here")


def main():
    st.set_page_config(
        page_title="AI Legal Assistant - Navigation",
        layout="wide",
        page_icon="⚖️",
        initial_sidebar_state="collapsed"
    )

    # Initialize session state
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'home'

    # Inject custom CSS
    st.markdown(inject_navigation_css(), unsafe_allow_html=True)

    # Check if we need to show an external app
    if st.session_state.current_page != 'home':
        show_external_app_page(load_and_run_module, show_fallback_interface)
        return

    show_home_page()


if __name__ == "__main__":
    main()