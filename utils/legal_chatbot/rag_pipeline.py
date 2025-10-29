from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from utils.legal_chatbot.memory_manager import MemoryManager
from vector_database import load_vector_store, process_user_pdf
import uuid
import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize LLM with error handling
try:
    llm_model = ChatGroq(
        model="llama-3.1-8b-instant",  # Faster and more reliable
        temperature=0.1,  # Lower temperature for factual responses
        groq_api_key=os.getenv("GROQ_API_KEY")  # Pass API key directly
    )
    GROQ_AVAILABLE = True
    print("✅ Groq initialized successfully")
except Exception as e:
    print(f"❌ Groq initialization failed: {e}")
    GROQ_AVAILABLE = False
    llm_model = None


def get_memory_manager():
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    return MemoryManager(session_id=st.session_state.session_id)


def get_context(documents):
    """Extract context from documents"""
    if not documents:
        return "No relevant legal documents found."
    return "\n\n".join([doc.page_content for doc in documents])


def retrieve_docs(query, custom_db=None):
    """Retrieve relevant documents from vector store"""
    try:
        db_to_use = custom_db if custom_db else load_vector_store()
        if not db_to_use:
            print("❌ No vector database available")
            return []
        return db_to_use.similarity_search(query, k=3)  # Get top 3 most relevant
    except Exception as e:
        print(f"❌ Error retrieving documents: {e}")
        return []


def answer_query(documents, query, memory_manager=None):
    """Answer query using Groq with enhanced error handling"""
    if not GROQ_AVAILABLE:
        return generate_fallback_response(documents, query)  # Use the defined function

    context = get_context(documents)

    try:
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

    except Exception as e:
        error_msg = f"❌ Error generating response: {str(e)}"
        print(error_msg)
        return generate_fallback_response(documents, query)  # Use the defined function


def process_user_query(uploaded_file, user_query, memory_manager):
    """Process query against user-uploaded document"""
    try:
        # Create vector store from user upload
        vector_store = process_user_pdf(uploaded_file)

        # Retrieve relevant documents
        retrieved_docs = retrieve_docs(user_query, vector_store)

        # Generate response
        response = answer_query_with_fallback(retrieved_docs, user_query, memory_manager)

        return response
    except Exception as e:
        return f"❌ Error processing your document: {str(e)}"


def _documents_are_relevant(documents, query):
    """Check if any document is actually relevant to the query"""
    if not documents:
        return False

    query_terms = set(query.lower().split())
    for doc in documents:
        content = doc.page_content.lower()
        # Check if any significant terms match
        matching_terms = [term for term in query_terms if len(term) > 3 and term in content]
        if len(matching_terms) >= 1:  # At least one significant term match
            return True
    return False


def should_use_fallback(retrieved_docs: list, rag_response: str) -> bool:
    """
    Determine if we should use fallback response
    """
    if not retrieved_docs:
        return True

    rag_response_lower = rag_response.lower()
    uncertainty_phrases = [
        "don't know", "not in the context", "i don't",
        "no information", "unable to answer", "cannot determine",
        "i couldn't", "isn't covered", "not present in context",
        "based on the provided documents", "the context doesn't"
    ]

    if any(phrase in rag_response_lower for phrase in uncertainty_phrases):
        return True

    if len(rag_response.split()) < 15:  # Too brief response
        return True

    return False


def generate_fallback_response(documents, query):
    """Generate a professional fallback response when AI fails"""
    if documents and _documents_are_relevant(documents, query):
        # We have relevant documents
        best_doc = documents[0].page_content
        preview = best_doc[:300] + "..." if len(best_doc) > 300 else best_doc

        return f"""📜 **Legal Information Found**

{preview}

💡 **Analysis**: I found relevant legal content in my knowledge base.

🔍 **Suggested**: For detailed legal interpretation and specific advice, please consult a qualified legal professional.

*Information sourced from analyzed legal documents*"""

    else:
        # No relevant documents found
        return f"""🔍 **Legal Research Result**

I couldn't find specific information about "{query}" in my current legal knowledge base.

**My expertise covers:**
• **Labour Laws**: Employment rights, wages, workplace regulations, disputes
• **Marriage Laws**: Hindu Marriage Act, Special Marriage Act, marriage procedures

**For comprehensive legal guidance:**
1. Consult the specific legal statutes directly
2. Speak with a qualified legal professional  
3. Check official government legal portals

📚 *My knowledge is based on specialized legal documents and may not cover all topics.*"""


def answer_query_with_fallback(documents, query, memory_manager=None):
    """Main function with proper fallback handling"""

    # First, try to get response from Groq
    rag_response = answer_query(documents, query, memory_manager)

    # Check if we need fallback
    if should_use_fallback(documents, rag_response):
        return generate_fallback_response(documents, query)

    return rag_response


def get_enhanced_prompt():
    """Enhanced prompt for legal responses"""
    template = """
You are an expert AI legal assistant specializing in Indian Labour Laws and Marriage Laws.

CONTEXT FROM LEGAL DOCUMENTS:
{context}

CONVERSATION HISTORY:
{chat_history}

USER QUESTION: {question}

IMPORTANT INSTRUCTIONS:
1. Provide ACCURATE, FACTUAL legal information based on the context
2. If the context contains relevant information, cite it specifically
3. If the context doesn't contain the answer, say "I don't have specific information on this in my knowledge base"
4. Be clear, practical, and professional
5. Mention relevant Indian laws/acts when possible
6. Never make up legal provisions

RESPONSE FORMAT:
- Start with direct answer
- Reference relevant laws if known
- Provide practical implications
- Include important cautions

ANSWER:
"""
    return PromptTemplate(
        template=template,
        input_variables=["chat_history", "context", "question"]
    )


# Emergency fallback for any remaining issues
def emergency_response(query):
    """Ultimate fallback for any errors"""
    return f"""⚖️ **Legal Assistant Response**

I'm currently experiencing technical difficulties with my AI service.

**Your Question:** "{query}"

**Suggested Action:** 
Please try again in a moment, or consult official legal resources for immediate assistance.

*My expertise includes Labour Laws and Marriage Laws in India.*"""


# Simple test function
def test_rag_pipeline():
    """Test the RAG pipeline"""
    print("🧪 Testing RAG Pipeline...")

    # Test with a simple query
    test_query = "What are labour laws?"

    try:
        vector_store = load_vector_store()
        if vector_store:
            docs = retrieve_docs(test_query, vector_store)
            print(f"📄 Retrieved {len(docs)} documents")

            response = answer_query_with_fallback(docs, test_query)
            print(f"🤖 Response: {response[:100]}...")
        else:
            print("❌ No vector store available")
    except Exception as e:
        print(f"❌ Test failed: {e}")


if __name__ == "__main__":
    test_rag_pipeline()