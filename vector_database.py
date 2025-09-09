# vector_database.py
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from utils.document_preprocessor import load_document, preprocess_documents
from config import *  # Import all constants from config


def load_pdf(file_path):
    documents = load_document(file_path)
    return preprocess_documents(documents)


def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_splitter.split_documents(documents)


def get_embedding_model():
    return OllamaEmbeddings(model=OLLAMA_MODEL_NAME)


def create_vector_store(text_chunks, db_path=PRETRAINED_DB_PATH):
    embeddings = get_embedding_model()
    faiss_db = FAISS.from_documents(text_chunks, embeddings)
    faiss_db.save_local(db_path)
    return faiss_db


def load_vector_store(db_path=PRETRAINED_DB_PATH):
    embeddings = get_embedding_model()
    try:
        return FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
    except:
        return None


def train_on_articles():
    all_documents = []

    for filename in os.listdir(KNOWLEDGE_BASE_DIR):
        if filename.lower().endswith('.pdf'):
            file_path = os.path.join(KNOWLEDGE_BASE_DIR, filename)
            try:
                documents = load_pdf(file_path)
                all_documents.extend(documents)
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                continue

    if not all_documents:
        raise ValueError("No valid documents could be processed")

    text_chunks = create_chunks(all_documents)
    return create_vector_store(text_chunks)


def process_user_pdf(uploaded_file):
    """Process a user-uploaded PDF and return temporary vector store"""
    file_path = os.path.join(USER_UPLOADS_DIR, uploaded_file.name)
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())

    documents = load_pdf(file_path)
    text_chunks = create_chunks(documents)
    embeddings = get_embedding_model()
    return FAISS.from_documents(text_chunks, embeddings)