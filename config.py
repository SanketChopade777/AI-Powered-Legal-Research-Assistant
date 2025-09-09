# config.py
import os

# Directory paths
KNOWLEDGE_BASE_DIR = 'knowledge_base/'
USER_UPLOADS_DIR = 'user_uploads/'
TRAINED_MODELS_DIR = 'trained_models/'
PRETRAINED_DB_PATH = "vectorstore/pretrained_db"
FAISS_DB_PATH = "vectorstore/db_faiss"
OLLAMA_MODEL_NAME = "deepseek-r1:1.5b"

# Ensure directories exist
os.makedirs(KNOWLEDGE_BASE_DIR, exist_ok=True)
os.makedirs(USER_UPLOADS_DIR, exist_ok=True)
os.makedirs(TRAINED_MODELS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(PRETRAINED_DB_PATH), exist_ok=True)