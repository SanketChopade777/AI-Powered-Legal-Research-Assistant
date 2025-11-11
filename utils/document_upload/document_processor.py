import os
import re
import pickle
from typing import Dict
import tempfile
import streamlit as st
from utils.legal_chatbot.document_preprocessor import load_document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import time


class DocumentProcessor:
    def __init__(self):
        self.supported_formats = ['.pdf', '.docx', '.doc', '.txt']
        self.embeddings = self._get_embeddings()
        self.user_uploads_dir = "user_uploads"
        os.makedirs(self.user_uploads_dir, exist_ok=True)

    def _get_embeddings(self):
        """Get embeddings using HuggingFace for better deployment"""
        try:
            # Use a lightweight, fast model for embeddings
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},  # Use CPU for compatibility
                encode_kwargs={'normalize_embeddings': True}
            )
            # st.success("✅ HuggingFace embeddings initialized successfully")
            return embeddings
        except Exception as e:
            st.error(f"❌ HuggingFace embeddings failed: {e}")
            # Ultimate fallback - try a different model
            try:
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",
                    model_kwargs={'device': 'cpu'}
                )
                st.info("✅ Fallback HuggingFace embeddings initialized")
                return embeddings
            except Exception as e2:
                st.error(f"❌ All embedding models failed: {e2}")
                raise Exception("No embedding model available")

    def extract_text(self, uploaded_file) -> str:
        """Extract text from uploaded file using your preprocessing system"""
        try:
            # Save uploaded file to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                temp_path = tmp_file.name

            # Show processing status
            st.info(f"📖 Processing document: {uploaded_file.name}")

            # Use your preprocessing system to load and process document
            documents = load_document(temp_path)

            if not documents:
                raise Exception("No content extracted from document")

            # Combine all document content
            full_text = "\n\n".join([doc.page_content for doc in documents])

            # Clean up temporary file
            os.unlink(temp_path)

            st.success(f"✅ Successfully extracted {len(full_text)} characters from document")
            return full_text

        except Exception as e:
            raise Exception(f"Error processing document: {str(e)}")

    def create_vector_store(self, text: str, document_name: str):
        """Create vector store from document text using HuggingFace embeddings"""
        try:
            from langchain.schema import Document

            # Show chunking progress
            with st.spinner("🔪 Splitting document into chunks..."):
                # Create document chunks
                documents = [Document(page_content=text, metadata={"source": document_name})]

                # Split into chunks optimized for HuggingFace
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,  # Slightly larger chunks for better context
                    chunk_overlap=150,
                    add_start_index=True
                )
                chunks = text_splitter.split_documents(documents)

                st.info(f"📄 Created {len(chunks)} text chunks for processing")
                time.sleep(1)  # Let user see the message

            # Show embedding creation progress
            with st.spinner("🧠 Creating embeddings with HuggingFace..."):
                # Create vector store with HuggingFace embeddings
                vector_store = FAISS.from_documents(chunks, self.embeddings)
                st.success("✅ Vector embeddings created successfully")
                time.sleep(1)

            return vector_store

        except Exception as e:
            raise Exception(f"Error creating vector store: {str(e)}")

    def save_processed_data(self, document_name: str, text: str, vector_store, analysis: Dict):
        """Save processed data to user_uploads folder"""
        try:
            # Create a safe filename
            safe_name = re.sub(r'[^\w\-_.]', '_', document_name)
            base_path = os.path.join(self.user_uploads_dir, safe_name)

            # Save text content
            text_path = f"{base_path}_text.txt"
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(text)

            # Save vector store (FAISS index)
            vector_path = f"{base_path}_faiss_index"
            vector_store.save_local(vector_path)

            # Save analysis
            analysis_path = f"{base_path}_analysis.pkl"
            with open(analysis_path, 'wb') as f:
                pickle.dump(analysis, f)

            st.success(f"💾 Document data saved to user_uploads folder")
            return {
                'text_path': text_path,
                'vector_path': vector_path,
                'analysis_path': analysis_path
            }

        except Exception as e:
            st.warning(f"⚠️ Could not save processed data: {e}")
            return None

    def load_processed_data(self, document_name: str):
        """Load previously processed data"""
        try:
            safe_name = re.sub(r'[^\w\-_.]', '_', document_name)
            base_path = os.path.join(self.user_uploads_dir, safe_name)

            # Check if files exist
            text_path = f"{base_path}_text.txt"
            vector_path = f"{base_path}_faiss_index"
            analysis_path = f"{base_path}_analysis.pkl"

            if not all(os.path.exists(p) for p in [text_path, vector_path, analysis_path]):
                return None

            # Load data
            with open(text_path, 'r', encoding='utf-8') as f:
                text = f.read()

            # Load FAISS index
            vector_store = FAISS.load_local(vector_path, self.embeddings, allow_dangerous_deserialization=True)

            with open(analysis_path, 'rb') as f:
                analysis = pickle.load(f)

            st.info("📂 Loaded previously processed document")
            return {
                'text': text,
                'vector_store': vector_store,
                'analysis': analysis
            }

        except Exception as e:
            st.warning(f"⚠️ Could not load processed data: {e}")
            return None

    def analyze_document_structure(self, text: str) -> Dict:
        """Analyze document structure and extract key information"""
        with st.spinner("🔍 Analyzing document structure..."):
            analysis = {
                'sections': [],
                'clauses': [],
                'parties': [],
                'dates': [],
                'monetary_values': [],
                'word_count': len(text.split()),
                'char_count': len(text),
                'estimated_pages': max(1, len(text) // 1500),  # Rough estimate
                'paragraphs': len([p for p in text.split('\n\n') if p.strip()])
            }

            # Extract sections (lines that might be section headers)
            lines = text.split('\n')
            for line in lines:
                line = line.strip()
                if line and len(line) < 100:  # Likely a section header
                    if re.match(r'^(ARTICLE|SECTION|CHAPTER|CLAUSE)\s+[IVXLCDM0-9]', line.upper()):
                        analysis['sections'].append(line)
                    elif line.isupper() or line.endswith(':'):
                        analysis['sections'].append(line)

            # Extract dates
            date_patterns = [
                r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
                r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b'
            ]
            for pattern in date_patterns:
                analysis['dates'].extend(re.findall(pattern, text, re.IGNORECASE))

            # Extract monetary values
            monetary_pattern = r'\$\s?\d+(?:,\d{3})*(?:\.\d{2})?|\d+(?:,\d{3})*(?:\.\d{2})?\s?(?:USD|dollars)'
            analysis['monetary_values'].extend(re.findall(monetary_pattern, text, re.IGNORECASE))

            return analysis

    def generate_quick_summary(self, text: str) -> str:
        """Generate a quick summary of the document"""
        # Extract first few paragraphs for quick preview
        paragraphs = [p for p in text.split('\n\n') if p.strip()]
        preview = "\n\n".join(paragraphs[:3])  # First 3 paragraphs

        # Basic analysis
        word_count = len(text.split())
        char_count = len(text)

        summary = f"""
🎉 **Document Processing Complete!**

📊 **Document Overview:**
- **Words:** {word_count:,} 
- **Characters:** {char_count:,}
- **Estimated Pages:** {max(1, word_count // 250)}
- **Paragraphs:** {len(paragraphs)}

📝 **Content Preview:**
{preview[:500]}{'...' if len(preview) > 500 else ''}

🚀 **Ready for Analysis!** You can now:
1. **Ask Questions** about specific content in the Q&A section below
2. **Run AI Analysis** for detailed insights using the options above

💡 **Tip:** Use the Q&A section below to ask specific questions about your document.
"""
        return summary