from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from utils.legal_chatbot.document_preprocessor import load_document
from config import *
import os


def get_embedding_model(use_ollama=False):
    """Get appropriate embedding model based on use case"""
    if use_ollama:
        try:
            # For user uploads - use Ollama (local, no internet needed)
            return OllamaEmbeddings(model=OLLAMA_MODEL)
        except Exception as e:
            print(f"❌ Ollama not available: {e}. Falling back to HuggingFace...")
            return get_huggingface_embeddings()
    else:
        # For pre-trained model - use HuggingFace (consistent with Colab)
        return get_huggingface_embeddings()


def get_huggingface_embeddings():
    """Get HuggingFace embeddings with error handling"""
    try:
        return HuggingFaceEmbeddings(
            model_name=HUGGINGFACE_MODEL,
            model_kwargs={'device': 'cpu'}
        )
    except Exception as e:
        print(f"❌ HuggingFace model not available: {e}")
        # Fallback to Ollama if HuggingFace fails
        return OllamaEmbeddings(model=OLLAMA_MODEL)


def load_vector_store(db_path=PRETRAINED_DB_PATH):
    """Load pre-trained FAISS index - use HuggingFace for consistency"""
    embeddings = get_huggingface_embeddings()
    try:
        vector_store = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
        print("✅ Pre-trained model loaded with HuggingFace embeddings")
        return vector_store
    except Exception as e:
        print(f"❌ Error loading vector store: {e}")
        return None


def process_user_pdf(uploaded_file):
    """Process user-uploaded PDF using Ollama (works offline)"""
    file_path = os.path.join(USER_UPLOADS_DIR, uploaded_file.name)

    # Save uploaded file
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())

    # Load and process document
    documents = load_document(file_path)
    if not documents:
        raise ValueError("No content extracted from uploaded file")

    # Create chunks
    text_chunks = create_chunks(documents)

    # Use Ollama for user uploads (works without internet)
    embeddings = get_embedding_model(use_ollama=True)

    print(f"✅ Processing user upload with {'Ollama' if 'Ollama' in str(type(embeddings)) else 'HuggingFace'}")
    return FAISS.from_documents(text_chunks, embeddings)


def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_splitter.split_documents(documents)


def train_on_articles():
    """Only for local training - use HuggingFace for consistency"""
    print("⚠️ Using Colab pre-trained model. Local training skipped.")
    return load_vector_store()