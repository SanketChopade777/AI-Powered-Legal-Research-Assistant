from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from utils.memory_manager import MemoryManager
from vector_database import load_vector_store, process_user_pdf
import uuid
import streamlit as st
from dotenv import load_dotenv
from utils.gemini_integration import GeminiIntegration

load_dotenv()

# Initialize LLM
llm_model = ChatGroq(model="deepseek-r1-distill-llama-70b")
gemini = GeminiIntegration()  # Initialize Gemini here


def get_memory_manager():
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    return MemoryManager(session_id=st.session_state.session_id)


def get_context(documents):
    return "\n\n".join([doc.page_content for doc in documents])


def retrieve_docs(query, custom_db=None):
    db_to_use = custom_db if custom_db else load_vector_store()
    if not db_to_use:
        raise ValueError("No vector database available")
    return db_to_use.similarity_search(query)


def answer_query(documents, query, memory_manager=None):
    context = get_context(documents)
    prompt = get_enhanced_prompt()
    chain = prompt | llm_model

    memory_vars = {}
    if memory_manager:
        memory = memory_manager.get_memory()
        memory_vars = {"chat_history": memory.get("chat_history", "")}

    response = chain.invoke({
        "question": query,
        "context": context,
        **memory_vars
    })

    if memory_manager:
        memory_manager.add_to_memory(query, response.content)

    return response.content


def process_user_query(uploaded_file, query, memory_manager=None):
    if not uploaded_file:
        raise ValueError("No file uploaded")

    if 'user_db' not in st.session_state:
        with st.spinner("Processing your document..."):
            st.session_state.user_db = process_user_pdf(uploaded_file)

    retrieved_docs = retrieve_docs(query, st.session_state.user_db)
    return answer_query(retrieved_docs, query, memory_manager)


def _documents_are_relevant(documents, query):
    """Check if any document is actually relevant to the query"""
    if not documents:
        return False

    query_terms = set(query.lower().split())
    for doc in documents:
        content = doc.page_content.lower()
        if any(term in content for term in query_terms if len(term) > 3):
            return True
    return False


def should_use_gemini(retrieved_docs: list, rag_response: str) -> tuple[bool, str]:
    """
    Determine if we should use Gemini as fallback
    """
    if not retrieved_docs:
        return (True, "No documents found")

    rag_response_lower = rag_response.lower()
    uncertainty_phrases = [
        "don't know", "not in the context", "i don't",
        "no information", "unable to answer", "cannot determine",
        "i couldn't", "isn't covered", "not present in context"
    ]

    if any(phrase in rag_response_lower for phrase in uncertainty_phrases):
        return (True, "Response indicates uncertainty about the answer")

    if len(rag_response.split()) < 10:
        return (True, "Response is too brief")

    return (False, "")


def answer_query_with_fallback(documents, query, memory_manager=None):
    # First check if documents are irrelevant
    if documents and not _documents_are_relevant(documents, query):
        use_gemini = True
        rag_response = "I couldn't find relevant information in the provided documents."
    else:
        context = get_context(documents) if documents else None
        rag_response = answer_query(documents, query, memory_manager)
        use_gemini, _ = should_use_gemini(documents, rag_response)

    # Handle Gemini response if needed
    if use_gemini and gemini.is_available():
        context = get_context(documents) if documents else None
        gemini_response = gemini.generate_response(query, context)

        if "non-legal question" in gemini_response.lower():
            return gemini_response
        return f"{rag_response}\n\nAdditional information:\n{gemini_response}"

    return rag_response


def get_enhanced_prompt():
    template = """
    You are an AI legal assistant. Use the following context to answer the question.
    If you don't know, say "I don't know based on the provided documents."

    Previous conversation:
    {chat_history}

    Legal Context:
    {context}

    Question: {question}
    Answer professionally, citing relevant articles when possible:
    """
    return PromptTemplate(
        template=template,
        input_variables=["chat_history", "context", "question"]
    )