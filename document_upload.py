import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import webbrowser


def inject_document_css():
    st.markdown("""
    <style>
    /* Main background - Dark theme */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #ffffff;
    }

    /* Main content area */
    .main .block-container {
        background: rgba(26, 26, 46, 0.9);
        border-radius: 20px;
        padding: 30px;
        margin: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.4);
        min-height: 80vh;
        border: 1px solid #4cc9f0;
    }

    /* Upload area styling */
    .upload-section {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 2px dashed #4cc9f0;
        border-radius: 20px;
        padding: 40px;
        text-align: center;
        margin: 20px 0;
        transition: all 0.3s ease;
    }

    .upload-section:hover {
        border-color: #f72585;
        background: linear-gradient(135deg, #1a1a2e 0%, #1e2a4a 100%);
    }

    .upload-icon {
        font-size: 4rem;
        color: #4cc9f0;
        margin-bottom: 20px;
    }

    /* Analysis results */
    .analysis-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 15px;
        padding: 20px;
        margin: 15px 0;
        border-left: 5px solid #f72585;
    }

    .highlight {
        background: linear-gradient(135deg, #f72585 0%, #b5179e 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)


def main():
    # st.set_page_config(
    #     page_title="LegalEase AI - Document Analysis",
    #     layout="wide",
    #     page_icon="📄",
    #     initial_sidebar_state="expanded"
    # )

    inject_document_css()

    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #4361ee 0%, #3a0ca3 100%); 
                   padding: 25px; border-radius: 20px; color: white; text-align: center; margin-bottom: 20px;'>
            <h1>📄 Document Analysis</h1>
            <p style='margin: 0;'>AI-Powered Legal Document Review</p>
        </div>
        """, unsafe_allow_html=True)

        # if st.button("🏠 Back to Main Menu", use_container_width=True):
        #     webbrowser.open_new_tab("http://localhost:8501")

        st.markdown("---")
        st.markdown("### 📊 Supported Formats")
        st.info("""
        - **PDF** Documents
        - **DOC/DOCX** Files  
        - **TXT** Files
        - **Images** (OCR)
        """)

        st.markdown("### 🔍 Analysis Features")
        st.info("""
        - Key Clause Identification
        - Risk Assessment
        - Compliance Checking
        - Summary Generation
        - Recommendation Engine
        """)

    # Main content
    st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <h1 style='color: #4cc9f0;'>📄 AI Document Analysis</h1>
        <p style='color: #a8b2d1;'>Upload your legal documents for instant AI-powered analysis and insights</p>
    </div>
    """, unsafe_allow_html=True)

    # Upload section
    st.markdown("""
    <div class='upload-section'>
        <div class='upload-icon'>📤</div>
        <h2>Upload Your Legal Document</h2>
        <p>Drag and drop your file here or click to browse</p>
        <p><small>Supported formats: PDF, DOC, DOCX, TXT (Max size: 25MB)</small></p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "docx", "doc", "txt"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        # File info
        file_details = {
            "Filename": uploaded_file.name,
            "File size": f"{uploaded_file.size / 1024 / 1024:.2f} MB",
            "File type": uploaded_file.type,
            "Uploaded on": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        st.success("✅ File uploaded successfully!")

        # Display file info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📄 File Name", uploaded_file.name)
        with col2:
            st.metric("📊 File Size", f"{uploaded_file.size / 1024 / 1024:.2f} MB")
        with col3:
            st.metric("🔍 Status", "Ready for Analysis")

        # Analysis options
        st.markdown("### 🎯 Analysis Options")
        analysis_types = st.multiselect(
            "Select analysis types:",
            ["Key Clause Extraction", "Risk Assessment", "Compliance Check",
             "Summary Generation", "Recommendation Engine"],
            default=["Key Clause Extraction", "Summary Generation"]
        )

        if st.button("🚀 Analyze Document", use_container_width=True):
            with st.spinner("🔍 Analyzing document with AI..."):
                # Simulate analysis process
                import time
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.02)
                    progress_bar.progress(i + 1)

                # Display results
                st.markdown("## 📊 Analysis Results")

                # Summary
                st.markdown("""
                <div class='analysis-card'>
                    <h3>📝 Executive Summary</h3>
                    <p>This document appears to be a <span class='highlight'>standard rental agreement</span> 
                    containing typical clauses for residential tenancy. The agreement follows standard legal 
                    frameworks with appropriate termination clauses and tenant protections.</p>
                </div>
                """, unsafe_allow_html=True)

                # Key findings
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("""
                    <div class='analysis-card'>
                        <h3>✅ Strengths</h3>
                        <ul>
                        <li>Clear termination clauses</li>
                        <li>Comprehensive tenant rights</li>
                        <li>Standard security deposit terms</li>
                        <li>Proper maintenance responsibilities</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    st.markdown("""
                    <div class='analysis-card'>
                        <h3>⚠️ Areas to Review</h3>
                        <ul>
                        <li>Ambiguous repair timelines</li>
                        <li>Vague subletting conditions</li>
                        <li>Unclear dispute resolution process</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)

                # Risk assessment
                st.markdown("""
                <div class='analysis-card'>
                    <h3>📈 Risk Assessment</h3>
                    <p>Overall Risk Level: <span class='highlight'>LOW TO MODERATE</span></p>
                    <p>This document presents standard legal risks associated with rental agreements. 
                    No major red flags detected, but recommend legal review for specific clauses.</p>
                </div>
                """, unsafe_allow_html=True)

                # Recommendations
                st.markdown("""
                <div class='analysis-card'>
                    <h3>💡 Recommendations</h3>
                    <ol>
                    <li>Clarify repair response timelines in Section 4.2</li>
                    <li>Define subletting conditions more precisely</li>
                    <li>Consider adding mediation clause for disputes</li>
                    <li>Review local tenancy law compliance</li>
                    </ol>
                </div>
                """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()