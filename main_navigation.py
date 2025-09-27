import sys
import streamlit as st
import importlib.util
import os


# Custom CSS for beautiful navigation page
def inject_navigation_css():
    st.markdown("""
    <style>
    /* Main background - Dark theme */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Navigation cards styling */
    
    /* Navigation cards with enhanced animations */
    .nav-card {
        background: rgba(26, 26, 46, 0.8);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 40px 30px;
        margin: 20px 0;
        box-shadow: 0 15px 35px rgba(0,0,0,0.5);
        border: 2px solid rgba(76, 201, 240, 0.3);
        transition: all 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        text-align: center;
        color: white;
        height: 320px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        position: relative;
        overflow: hidden;
        opacity: 0;
        transform: translateY(30px);
        animation: fadeInUp 0.8s ease forwards;
    }

    .nav-card:nth-child(1) { animation-delay: 0.2s; }
    .nav-card:nth-child(2) { animation-delay: 0.4s; }
    .nav-card:nth-child(3) { animation-delay: 0.6s; }

    @keyframes fadeInUp {
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .nav-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(76, 201, 240, 0.2), transparent);
        transition: left 0.6s;
    }

    .nav-card:hover::before {
        left: 100%;
    }

    .nav-card:hover {
        transform: translateY(-10px) scale(1.02);
        box-shadow: 0 25px 50px rgba(76, 201, 240, 0.3);
        border-color: #f72585;
    }

    
    .nav-icon {
        font-size: 4rem;
        margin-bottom: 20px;
        background: linear-gradient(135deg, #4cc9f0 0%, #f72585 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        filter: drop-shadow(0 5px 15px rgba(76, 201, 240, 0.5));
        transition: all 0.3s ease;
    }

    .nav-card:hover .nav-icon {
        transform: scale(1.1) rotate(5deg);
        filter: drop-shadow(0 8px 25px rgba(247, 37, 133, 0.6));
    }
    
    .nav-title {
        font-size: 1.8rem;
        font-weight: bold;
        margin-bottom: 15px;
        background: linear-gradient(135deg, #94d2e5 0%, #ffffff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .nav-description {
        font-size: 1rem;
        opacity: 0.9;
        margin-bottom: 20px;
        line-height: 1.5;
    }

    .stButton>button {
        background: linear-gradient(135deg, #f72585 0%, #b5179e 100%);
        color: white;
        border: none;
        padding: 12px 10px;
        border-radius: 25px;
        font-weight: bold;
        font-size: 1rem;
        transition: all 0.3s ease;
        width: 40%;
    }

    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 8px 25px rgba(247, 37, 133, 0.4);
        background: linear-gradient(135deg, #b5179e 0%, #7209b7 60%);
    }

    /* Header styling */
    .main-header {
        text-align: center;
        padding: 40px 20px;
        margin-bottom: 30px;
    }

    .main-title {
        font-size: 4rem;
        font-weight: bold;
        color: #4cc9f0;
        margin-bottom: 15px;
    }

    .main-subtitle {
        font-size: 1.4rem;
        color: #a8b2d1;
        max-width: 600px;
        margin: 0 auto;
    }

    /* Footer styling */
    .footer {
        text-align: center;
        color: #a8b2d1;
        padding: 40px 20px;
        margin-top: 60px;
        border-top: 1px solid rgba(76, 201, 240, 0.3);
        background: rgba(10, 10, 20, 0.5);
        backdrop-filter: blur(10px);
    }
    
     /* Loading spinner */
    .loading-spinner {
        display: inline-block;
        width: 50px;
        height: 50px;
        border: 5px solid rgba(76, 201, 240, 0.3);
        border-radius: 50%;
        border-top-color: #4cc9f0;
        animation: spin 1s ease-in-out infinite;
        margin: 20px auto;
    }

    @keyframes spin {
        to { transform: rotate(360deg); }
    }

    /* Feature cards animation */
    .feature-card {
        background: rgba(26, 26, 46, 0.6);
        border-radius: 15px;
        padding: 30px;
        margin: 15px 0;
        border-left: 4px solid #4cc9f0;
        transition: all 0.3s ease;
        opacity: 0;
        transform: translateY(30px);
        animation: fadeInUp 0.8s ease forwards;
    }

    .feature-card:nth-child(1) { animation-delay: 1s; }
    .feature-card:nth-child(2) { animation-delay: 1.2s; }
    .feature-card:nth-child(3) { animation-delay: 1.4s; }

    .feature-card:hover {
        background: rgba(26, 26, 46, 0.9);
        transform: translateY(-5px);
        border-left-color: #f72585;
    }
    </style>
    """, unsafe_allow_html=True)


def run_external_app(app_file):
    """Navigate to external app"""
    st.session_state.current_page = app_file
    st.rerun()


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
        page_title="LegalEase AI - Navigation",
        layout="wide",
        page_icon="⚖️",
        initial_sidebar_state="collapsed"
    )

    # Initialize session state
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'home'

    # Inject custom CSS
    inject_navigation_css()

    # Check if we need to show an external app
    if st.session_state.current_page != 'home':
        show_external_app_page()
        return

    show_home_page()


def show_home_page():
    """Show the main navigation page"""
    st.markdown("""
    <div class="main-header">
        <div class="main-title">⚖️ LegalEase AI</div>
        <div class="main-subtitle">Your comprehensive legal assistance platform powered by artificial intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    # Create three columns for navigation cards
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="nav-card">
            <div class="nav-icon">📄</div>
            <div class="nav-title">Document Analysis</div>
            <div class="nav-description">
                Upload and analyze legal documents with AI-powered insights. 
                Get instant summaries, identify key clauses, and understand complex legal language.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Button below the card
        if st.button("Analyze Documents", key="doc_btn", use_container_width=True):
            run_external_app('document_upload.py')

    with col2:
        st.markdown("""
        <div class="nav-card">
            <div class="nav-icon">🤖</div>
            <div class="nav-title">Legal Chatbot</div>
            <div class="nav-description">
                Chat with our AI legal assistant trained on comprehensive legal knowledge. 
                Get answers to your legal questions instantly.
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Start Chatting", key="chat_btn", use_container_width=True):
            run_external_app('legal_chatbot.py')

    with col3:
        st.markdown("""
        <div class="nav-card">
            <div class="nav-icon">👨‍💼</div>
            <div class="nav-title">Lawyer Finder</div>
            <div class="nav-description">
                Find qualified lawyers based on your specific needs. 
                Filter by expertise, location, ratings, and availability.
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Find Lawyers", key="lawyer_btn", use_container_width=True):
            run_external_app('lawyer_finder.py')

 # Additional information section
    st.markdown("---")
    st.markdown("""
       <div style="text-align: center; padding: 40px 20px;">
           <h2 style="color: #4cc9f0; margin-bottom: 20px;">How It Works 🤔</h2>
       </div>
       """, unsafe_allow_html=True)

    features_col1, features_col2, features_col3 = st.columns(3)

    with features_col1:
        st.markdown("""
           <div class="feature-card">
               <h3 style="color: #f72585; margin-bottom: 15px;">📝 Document Analysis</h3>
               <p style="color: #e0e0ff; line-height: 1.6;">
                   Upload contracts, agreements, or legal documents for instant AI-powered analysis and insights.
               </p>
           </div>
           """, unsafe_allow_html=True)

    with features_col2:
        st.markdown("""
           <div class="feature-card">
               <h3 style="color: #f72585; margin-bottom: 15px;">💬 Smart Chat</h3>
               <p style="color: #e0e0ff; line-height: 1.6;">
                   Get answers to complex legal questions from our trained AI assistant.
               </p>
           </div>
           """, unsafe_allow_html=True)

    with features_col3:
        st.markdown("""
           <div class="feature-card">
               <h3 style="color: #f72585; margin-bottom: 15px;">🔍 Expert Matching</h3>
               <p style="color: #e0e0ff; line-height: 1.6;">
                   Connect with verified legal professionals tailored to your specific needs.
               </p>
           </div>
           """, unsafe_allow_html=True)

    # Statistics section
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; padding: 40px 20px;">
            <h2 style="color: #4cc9f0; margin-bottom: 30px; font-size: 2.5rem;">📊 Platform Statistics</h2>
        </div>
        """, unsafe_allow_html=True)

    stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)

    with stats_col1:
        st.markdown("""
            <div style="text-align: center; padding: 20px;">
                <div style="font-size: 3rem; color: #4cc9f0; font-weight: bold;">500+</div>
                <div style="color: #a8b2d1; font-size: 1.1rem;">Legal Documents Analyzed</div>
            </div>
            """, unsafe_allow_html=True)

    with stats_col2:
        st.markdown("""
            <div style="text-align: center; padding: 20px;">
                <div style="font-size: 3rem; color: #f72585; font-weight: bold;">1K+</div>
                <div style="color: #a8b2d1; font-size: 1.1rem;">Questions Answered</div>
            </div>
            """, unsafe_allow_html=True)

    with stats_col3:
        st.markdown("""
            <div style="text-align: center; padding: 20px;">
                <div style="font-size: 3rem; color: #4cc9f0; font-weight: bold;">200+</div>
                <div style="color: #a8b2d1; font-size: 1.1rem;">Verified Lawyers</div>
            </div>
            """, unsafe_allow_html=True)

    with stats_col4:
        st.markdown("""
            <div style="text-align: center; padding: 20px;">
                <div style="font-size: 3rem; color: #f72585; font-weight: bold;">99%</div>
                <div style="color: #a8b2d1; font-size: 1.1rem;">User Satisfaction</div>
            </div>
            """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div class="footer">
        <p>⚖️ LegalEase AI v2.0 | Built with Streamlit & Advanced AI Technologies</p>
        <p>🔒 Your data is secure and confidential</p>
    </div>
    """, unsafe_allow_html=True)


def show_external_app_page():
    """Show the external app with a back button"""

    # Back button at top
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("← Back to Home", key="back_btn"):
            st.session_state.current_page = 'home'
            st.rerun()

    # Run the external app
    app_file = st.session_state.current_page

    # Check if file exists
    if not os.path.exists(app_file):
        st.error(f"File {app_file} not found! Please make sure it exists in the same directory.")
        show_fallback_interface(app_file)
        return

    try:
        load_and_run_module(app_file)
    except Exception as e:
        st.error(f"Error running {app_file}: {str(e)}")
        show_fallback_interface(app_file)

# Fallback functions
def show_chatbot_fallback():
    st.markdown("## 🤖 Legal Chatbot (Fallback Mode)")
    st.info("The chatbot module is not available. Here's a simple chat interface:")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a legal question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response = "I'm a fallback chatbot. In the full version, I would provide legal insights based on your question."
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})


def show_document_fallback():
    st.markdown("## 📄 Document Analysis (Fallback Mode)")
    st.info("The document analysis module is not available. Here's a simple upload interface:")

    uploaded_file = st.file_uploader("Upload legal document", type=["pdf", "docx", "txt"])
    if uploaded_file:
        st.success(f"Uploaded: {uploaded_file.name}")
        if st.button("Analyze Document"):
            with st.spinner("Analyzing document..."):
                import time
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)

                st.success("Analysis complete!")
                st.json({
                    "document_type": "Legal Contract",
                    "key_clauses": ["Termination Clause", "Liability Clause"],
                    "risk_level": "Low",
                    "recommendations": ["Review section 4.2", "Consult for clause 7.1"]
                })


def show_lawyer_fallback():
    st.markdown("## 👨‍💼 Lawyer Finder (Fallback Mode)")
    st.info("The lawyer finder module is not available. Here's a simple search interface:")

    col1, col2 = st.columns(2)
    with col1:
        specialty = st.selectbox("Specialty", ["Corporate Law", "Criminal Defense", "Family Law", "Real Estate"])
    with col2:
        location = st.selectbox("Location", ["New York", "California", "Texas", "Florida"])

    if st.button("Search Lawyers"):
        lawyers = [
            {"name": "Dr. Sarah Johnson", "specialty": "Corporate Law", "rating": 4.8, "experience": "15 years"},
            {"name": "Robert Chen", "specialty": "Criminal Defense", "rating": 4.6, "experience": "8 years"},
            {"name": "Maria Rodriguez", "specialty": "Family Law", "rating": 4.9, "experience": "12 years"}
        ]

        for lawyer in lawyers:
            with st.expander(f"👤 {lawyer['name']} - ⭐ {lawyer['rating']}"):
                st.write(f"**Specialty:** {lawyer['specialty']}")
                st.write(f"**Experience:** {lawyer['experience']}")
                st.button("Contact Lawyer", key=f"contact_{lawyer['name']}")


if __name__ == "__main__":
    main()