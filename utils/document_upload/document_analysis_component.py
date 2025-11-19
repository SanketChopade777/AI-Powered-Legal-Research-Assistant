import streamlit as st
from utils.document_upload.document_processor import DocumentProcessor
from utils.document_upload.document_rag_analyzer import DocumentRAGAnalyzer
import time
from datetime import datetime


class DocumentAnalysisComponent:
    def __init__(self):
        self.processor = DocumentProcessor()
        self.rag_analyzer = DocumentRAGAnalyzer()

        # Initialize session state
        self._init_session_state()

    def _init_session_state(self):
        """Initialize session state variables"""
        if 'document_text' not in st.session_state:
            st.session_state.document_text = None
        if 'document_analysis' not in st.session_state:
            st.session_state.document_analysis = None
        if 'uploaded_file_name' not in st.session_state:
            st.session_state.uploaded_file_name = None
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = None
        if 'analysis_history' not in st.session_state:
            st.session_state.analysis_history = []
        if 'vector_store' not in st.session_state:
            st.session_state.vector_store = None
        if 'document_processed' not in st.session_state:
            st.session_state.document_processed = False

    def inject_document_css(self):
        st.markdown("""
        <style>
        /* Main background - Dark theme */
        .stApp {
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
            border: 1px solid rgba(76, 201, 240, 0.3);
        }

        .quick-summary {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            border-radius: 15px;
            padding: 25px;
            margin: 20px 0;
            border: 2px solid #4cc9f0;
        }

       /* User message - Deep Royal Blue & Purple */
        .user-message {
            background: linear-gradient(135deg, #182c61 0%, #301934 100%); /* Deep Royal Blue to Dark Plum */
            color: white; /* High contrast text */
            padding: 15px 20px;
            border-radius: 18px 18px 5px 18px;
            margin: 10px 0;
            max-width: 80%;
            margin-left: auto;
            border: 1px solid rgba(24, 44, 97, 0.4); /* Border color based on the start color */
        }
        
        /* Assistant message - Deep Forest Green & Teal */
        .assistant-message {
            background: linear-gradient(135deg, #004d40 0%, #00796b 100%); /* Deep Forest Green to Rich Teal */
            color: white; /* High contrast text */
            padding: 15px 20px;
            border-radius: 18px 18px 18px 5px;
            margin: 10px 0;
            max-width: 80%;
            margin-right: auto;
            border: 1px solid rgba(0, 77, 64, 0.4); /* Border color based on the start color */
        }

        /* Status indicators */
        .status-success {
            background: linear-gradient(135deg, #4ade80 0%, #16a34a 100%);
            color: white;
            padding: 10px 15px;
            border-radius: 10px;
            margin: 5px 0;
        }

        .status-processing {
            background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
            color: white;
            padding: 10px 15px;
            border-radius: 10px;
            margin: 5px 0;
        }
        </style>
        """, unsafe_allow_html=True)

    def render_back_button(self):
        """Render the back button to return to home"""
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if st.button("← Back to Home", key="doc_back_btn", use_container_width=True):
                if 'current_page' in st.session_state:
                    st.session_state.current_page = 'home'
                    st.rerun()

    def render_sidebar(self):
        """Render the sidebar content"""
        with st.sidebar:
            st.markdown("""
            <div style='background: linear-gradient(135deg, #4361ee 0%, #3a0ca3 100%); 
                       padding: 25px; border-radius: 20px; color: white; text-align: center; margin-bottom: 20px;'>
                <h1>📄 Document Analysis</h1>
                <p style='margin: 0;'>AI-Powered Legal Document Review</p>
            </div>
            """, unsafe_allow_html=True)

            # API Status
            st.markdown("### 🔌 System Status")
            if self.rag_analyzer.is_configured():
                st.markdown('<div class="status-success">✅ Groq API Connected (llama-3.1-8b-instant)</div>',
                            unsafe_allow_html=True)
            else:
                st.error("❌ Groq API Not Configured - Check your .env file")

            st.markdown("### 🔍 Processing Engine")
            st.info("""
            **Cloud-Ready Processing:**
            - HuggingFace Sentence Transformers
            - Fast & reliable embeddings
            - No local dependencies required
            - Perfect for deployment
            """)

            st.markdown("---")
            st.markdown("### 📊 Supported Formats")
            st.info("""
            - **PDF** Documents (text & scanned)
            - **DOC/DOCX** Files  
            - **TXT** Files
            - **Max Size:** 25MB (recommended: under 50 pages)
            """)

            # Document statistics
            if st.session_state.document_processed:
                st.markdown("---")
                st.markdown("### 📈 Document Stats")
                text = st.session_state.document_text
                analysis = st.session_state.document_analysis
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("📝 Words", f"{analysis['word_count']:,}")
                with col2:
                    st.metric("📄 Est. Pages", analysis['estimated_pages'])

                if analysis['sections']:
                    st.markdown("### 📑 Detected Sections")
                    with st.expander("View sections"):
                        for section in analysis['sections'][:8]:
                            st.write(f"• {section}")

    def render_upload_section(self):
        """Render the file upload section"""
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
            # Check if it's a new file
            current_file = getattr(st.session_state, 'current_file', None)
            if current_file != uploaded_file.name:
                st.session_state.current_file = uploaded_file.name
                self.process_uploaded_file(uploaded_file)

    def process_uploaded_file(self, uploaded_file):
        """Process the uploaded file with proper progress tracking"""
        try:
            # Check if document was already processed
            cached_data = self.processor.load_processed_data(uploaded_file.name)
            if cached_data:
                st.session_state.document_text = cached_data['text']
                st.session_state.document_analysis = cached_data['analysis']
                st.session_state.vector_store = cached_data['vector_store']
                st.session_state.uploaded_file_name = uploaded_file.name
                st.session_state.chat_history = []
                st.session_state.analysis_results = None
                st.session_state.document_processed = True

                st.success("✅ Loaded previously processed document!")
                return

            # Create a progress container
            progress_container = st.container()

            with progress_container:
                st.markdown("### 🔄 Processing Your Document...")

                # Step 1: Extract text
                with st.spinner("📖 Step 1/4: Extracting text from document..."):
                    document_text = self.processor.extract_text(uploaded_file)
                    time.sleep(1)  # Visual feedback

                # Step 2: Analyze structure
                with st.spinner("🔍 Step 2/4: Analyzing document structure..."):
                    document_analysis = self.processor.analyze_document_structure(document_text)
                    time.sleep(1)

                # Step 3: Create vector store
                with st.spinner("🧠 Step 3/4: Creating embeddings with HuggingFace..."):
                    vector_store = self.processor.create_vector_store(document_text, uploaded_file.name)
                    time.sleep(1)

                # Step 4: Save processed data
                with st.spinner("💾 Step 4/4: Saving processed data..."):
                    save_paths = self.processor.save_processed_data(
                        uploaded_file.name, document_text, vector_store, document_analysis
                    )
                    time.sleep(1)

            # Store in session state
            st.session_state.document_text = document_text
            st.session_state.document_analysis = document_analysis
            st.session_state.uploaded_file_name = uploaded_file.name
            st.session_state.vector_store = vector_store
            st.session_state.chat_history = []
            st.session_state.analysis_results = None
            st.session_state.document_processed = True

            # Show quick summary
            quick_summary = self.processor.generate_quick_summary(document_text)
            st.markdown(f"""
            <div class='quick-summary'>
                {quick_summary}
            </div>
            """, unsafe_allow_html=True)

            # Show document stats
            self._show_document_stats(document_analysis)

        except Exception as e:
            st.error(f"❌ Error processing document: {str(e)}")

    def _show_document_stats(self, analysis):
        """Show detailed document statistics"""
        with st.expander("📊 Detailed Document Statistics", expanded=True):
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("📝 Words", f"{analysis['word_count']:,}")
            with col2:
                st.metric("🔤 Characters", f"{analysis['char_count']:,}")
            with col3:
                st.metric("📄 Est. Pages", analysis['estimated_pages'])
            with col4:
                st.metric("📑 Paragraphs", analysis['paragraphs'])

            if analysis['sections']:
                st.markdown("**📋 Detected Sections:**")
                for i, section in enumerate(analysis['sections'][:10]):
                    st.write(f"{i + 1}. {section}")
                if len(analysis['sections']) > 10:
                    st.info(f"... and {len(analysis['sections']) - 10} more sections")

            if analysis['dates']:
                st.markdown(f"**📅 Dates Found:** {len(analysis['dates'])}")

            if analysis['monetary_values']:
                st.markdown(f"**💰 Monetary Values:** {len(analysis['monetary_values'])}")

    def render_analysis_options(self):
        """Render analysis options"""
        if not st.session_state.document_processed:
            return

        st.markdown("### 🎯 Advanced Analysis Options")

        analysis_types = st.multiselect(
            "Select analysis types:",
            ["Summary", "Risk Assessment", "Key Clauses", "Compliance Check", "Full Analysis"],
            default=["Summary"],
            key="analysis_types_select"
        )

        if st.button("🚀 Run AI Analysis", use_container_width=True, type="primary"):
            if not self.rag_analyzer.is_configured():
                st.error("❌ Groq API not configured. Please check your .env file.")
                return
            self.perform_analysis(analysis_types)

    def perform_analysis(self, analysis_types):
        """Perform document analysis using RAG approach"""
        progress_bar = st.progress(0)
        status_text = st.empty()

        analysis_results = {}
        total_analyses = len(analysis_types)

        for i, analysis_type in enumerate(analysis_types):
            progress = int((i / total_analyses) * 100)
            progress_bar.progress(progress)
            status_text.text(f"🔍 Performing {analysis_type} analysis... ({i + 1}/{total_analyses})")

            try:
                if analysis_type == "Summary":
                    result = self.rag_analyzer.generate_summary(
                        st.session_state.document_text,
                        st.session_state.uploaded_file_name
                    )
                else:
                    result = self.rag_analyzer.analyze_document_comprehensive(
                        st.session_state.document_text,
                        analysis_type.lower(),
                        st.session_state.uploaded_file_name
                    )

                analysis_results[analysis_type] = result

                # Add to analysis history
                st.session_state.analysis_history.append({
                    'type': analysis_type,
                    'timestamp': datetime.now().strftime("%H:%M:%S"),
                    'file': st.session_state.uploaded_file_name
                })

            except Exception as e:
                st.error(f"❌ Error in {analysis_type} analysis: {str(e)}")
                analysis_results[analysis_type] = {"error": str(e)}

            time.sleep(1)

        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")
        time.sleep(1)
        status_text.empty()

        st.session_state.analysis_results = analysis_results
        self.render_analysis_results(analysis_results)

    def render_analysis_results(self, analysis_results):
        """Render the analysis results"""
        st.markdown("## 📊 AI Analysis Results")

        for analysis_type, result in analysis_results.items():
            if "error" in result:
                st.error(f"❌ {analysis_type} Analysis Failed: {result['error']}")
                continue

            with st.expander(f"🎯 {analysis_type.upper()} ANALYSIS", expanded=True):
                st.markdown(f"""
                <div class='analysis-card'>
                    <div style='white-space: pre-wrap; line-height: 1.6;'>{result.get('content', 'No content available')}</div>
                </div>
                """, unsafe_allow_html=True)

    def render_qa_section(self):
        """Render the Q&A section - clean and simple"""
        if not st.session_state.get("document_processed", False):
            return

        # Initialize session state vars if missing
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if "question_input" not in st.session_state:
            st.session_state.question_input = ""

        st.markdown("---")

        # Simple header
        st.markdown("## 💬 Ask Questions About Your Document")
        st.info("Get instant answers about any part of your uploaded document")

        # Display chat history
        chat_container = st.container()
        with chat_container:
            if not st.session_state.chat_history:
                st.info("💡 Start by asking a question about your document in the box below.")
            else:
                for message in st.session_state.chat_history:
                    if message['role'] == 'user':
                        st.markdown(f"""
                        <div class='user-message'>
                            <strong>You:</strong><br>{message['content']}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class='assistant-message'>
                            <strong>Assistant:</strong><br>{message['content']}
                        </div>
                        """, unsafe_allow_html=True)

        # Question input with form for auto-clear
        with st.form(key="qa_form", clear_on_submit=True):
            question = st.text_area(
                "Type your question here:",
                placeholder="Ask anything about your document...",
                key="question_input",
                height=100,
                label_visibility="collapsed"
            )

            col1, col2 = st.columns([1, 1])

            with col1:
                submit_question = st.form_submit_button(
                    "🔍 Ask Question",
                    use_container_width=True,
                    type="primary"
                )

            with col2:
                submit_clear = st.form_submit_button(
                    "🗑️ Clear Chat",
                    use_container_width=True
                )

        # Handle form submissions
        if submit_question and question.strip():
            with st.spinner("🤔 Searching document for answers..."):
                # Add user question to chat
                st.session_state.chat_history.append({
                    'role': 'user',
                    'content': question,
                    'timestamp': datetime.now().strftime("%H:%M:%S")
                })

                try:
                    answer = self.rag_analyzer.answer_question(
                        st.session_state.document_text,
                        question,
                        st.session_state.uploaded_file_name
                    )

                    # Add assistant answer
                    st.session_state.chat_history.append({
                        'role': 'assistant',
                        'content': answer,
                        'timestamp': datetime.now().strftime("%H:%M:%S")
                    })

                except Exception as e:
                    error_msg = f"❌ Error answering question: {str(e)}"
                    st.session_state.chat_history.append({
                        'role': 'assistant',
                        'content': error_msg,
                        'timestamp': datetime.now().strftime("%H:%M:%S")
                    })

            # Form will auto-clear due to clear_on_submit=True
            st.rerun()

        elif submit_clear:
            st.session_state.chat_history = []
            st.rerun()

    def main(self):
        """Main function to run the document analysis component"""
        self.inject_document_css()

        # Add back button at the top
        self.render_back_button()

        # Main content
        st.markdown("""
        <div style='text-align: center; padding: 20px;'>
            <h1 style='color: #4cc9f0; font-size: 2.5rem;'>📄 AI Document Analysis</h1>
            <p style='color: #a8b2d1; font-size: 1.2rem;'>Upload legal documents for AI-powered analysis and Q&A</p>
        </div>
        """, unsafe_allow_html=True)

        # Render components
        self.render_sidebar()
        self.render_upload_section()

        # Show analysis options and Q&A when document is processed
        if st.session_state.document_processed:
            self.render_analysis_options()

            # Show analysis results if available
            # if st.session_state.analysis_results:
            #     self.render_analysis_results(st.session_state.analysis_results)

            # Always show Q&A section
            self.render_qa_section()