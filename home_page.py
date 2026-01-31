import streamlit as st
from navigation_utils import run_external_app

def show_home_page():
    """Show the main navigation page"""
    st.markdown("""
    <div class="main-header">
        <div class="main-title">⚖️ AI Legal Assistant</div>
        <div class="main-subtitle">Your comprehensive legal assistance platform powered by artificial intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    # Create three columns for navigation cards
    col1, col2, col3 = st.columns(3)

    with col1:
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

    with col2:
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
                       <h3 style="color: #f72585; margin-bottom: 15px;">💬 Smart Chat</h3>
                       <p style="color: #e0e0ff; line-height: 1.6;">
                           Get answers to complex legal questions from our trained AI assistant.
                       </p>
                   </div>
                   """, unsafe_allow_html=True)

    with features_col2:
        st.markdown("""
           <div class="feature-card">
               <h3 style="color: #f72585; margin-bottom: 15px;">📝 Document Analysis</h3>
               <p style="color: #e0e0ff; line-height: 1.6;">
                   Upload contracts, agreements, or legal documents for instant AI-powered analysis and insights.
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
                <div style="font-size: 3rem; color: #4cc9f0; font-weight: bold;">10+</div>
                <div style="color: #a8b2d1; font-size: 1.1rem;">Legal Documents Analyzed</div>
            </div>
            """, unsafe_allow_html=True)

    with stats_col2:
        st.markdown("""
            <div style="text-align: center; padding: 20px;">
                <div style="font-size: 3rem; color: #f72585; font-weight: bold;">100+</div>
                <div style="color: #a8b2d1; font-size: 1.1rem;">Questions Answered</div>
            </div>
            """, unsafe_allow_html=True)

    with stats_col3:
        st.markdown("""
            <div style="text-align: center; padding: 20px;">
                <div style="font-size: 3rem; color: #4cc9f0; font-weight: bold;">100+</div>
                <div style="color: #a8b2d1; font-size: 1.1rem;">Verified Lawyers</div>
            </div>
            """, unsafe_allow_html=True)

    with stats_col4:
        st.markdown("""
            <div style="text-align: center; padding: 20px;">
                <div style="font-size: 3rem; color: #f72585; font-weight: bold;">90%</div>
                <div style="color: #a8b2d1; font-size: 1.1rem;">User Satisfaction</div>
            </div>
            """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div class="footer">
        <p>⚖️ AI Legal Assistant v2.0 | Built with Streamlit & Advanced AI Technologies</p>
        <p>🔒 Your data is secure and confidential</p>
    </div>
    """, unsafe_allow_html=True)