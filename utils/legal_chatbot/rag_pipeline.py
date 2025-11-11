from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from utils.legal_chatbot.memory_manager import MemoryManager
from vector_database import load_vector_store, process_user_pdf
from utils.legal_chatbot.token_manager import TokenManager
import uuid
import streamlit as st
from dotenv import load_dotenv
import os
import traceback

load_dotenv()

# Initialize components
token_manager = TokenManager()

# Initialize LLM with error handling
try:
    llm_model = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.1,
        groq_api_key=os.getenv("GROQ_API_KEY"),
        max_tokens=500
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


def safe_get_context(documents, query):
    """Safely extract and optimize context from documents"""
    try:
        if not documents:
            return "No relevant legal documents found."

        return token_manager.optimize_context(documents, query)
    except Exception as e:
        print(f"❌ Error in safe_get_context: {e}")
        # Fallback: simple concatenation
        if documents:
            contents = []
            for doc in documents[:3]:
                content = token_manager.safe_string_convert(doc)
                if content.strip():
                    contents.append(content[:500])  # Limit each doc
            return "\n\n".join(contents) if contents else "No content available."
        return "No content available."


def retrieve_docs(query, custom_db=None):
    """Retrieve relevant documents from vector store"""
    try:
        db_to_use = custom_db if custom_db else load_vector_store()
        if not db_to_use:
            print("❌ No vector database available")
            return []

        # Ensure query is string
        query_str = token_manager.safe_string_convert(query)
        retrieved_docs = db_to_use.similarity_search(query_str, k=5)
        print(f"📄 Retrieved {len(retrieved_docs)} documents")
        return retrieved_docs

    except Exception as e:
        print(f"❌ Error retrieving documents: {e}")
        return []


def safe_invoke_chain(chain, inputs):
    """Safely invoke the chain with proper error handling"""
    try:
        return chain.invoke(inputs)
    except Exception as e:
        print(f"❌ Error in chain invocation: {e}")
        print(f"Inputs type: {type(inputs)}")
        print(f"Inputs keys: {inputs.keys() if hasattr(inputs, 'keys') else 'No keys'}")
        raise e


def answer_query(documents, query, memory_manager=None):
    """Answer query using Groq with enhanced error handling"""
    if not GROQ_AVAILABLE:
        return generate_fallback_response(documents, query)

    try:
        # Get optimized context with token management
        context = safe_get_context(documents, query)

        # Count tokens for debugging
        context_tokens = token_manager.count_tokens(context)
        print(f"🔢 Using {context_tokens} tokens for context")

        prompt = get_enhanced_prompt()
        chain = prompt | llm_model

        memory_vars = {}
        if memory_manager:
            try:
                memory = memory_manager.get_memory()
                # Safely get chat history
                chat_history = memory.get("chat_history", "")
                if not isinstance(chat_history, str):
                    chat_history = str(chat_history)

                # Truncate chat history if too long
                if token_manager.count_tokens(chat_history) > 500:
                    chat_history = token_manager.truncate_text(chat_history, 500)
                memory_vars = {"chat_history": chat_history}
            except Exception as e:
                print(f"❌ Error with memory manager: {e}")
                memory_vars = {"chat_history": ""}

        # Prepare inputs safely
        inputs = {
            "question": token_manager.safe_string_convert(query),
            "context": context,
            **memory_vars
        }

        # Validate all inputs are strings
        for key, value in inputs.items():
            if not isinstance(value, str):
                inputs[key] = token_manager.safe_string_convert(value)
                print(f"⚠️ Converted {key} to string: {type(value)} -> str")

        response = safe_invoke_chain(chain, inputs)

        if memory_manager:
            try:
                memory_manager.add_to_memory(query, response.content)
            except Exception as e:
                print(f"❌ Error adding to memory: {e}")

        return response.content

    except Exception as e:
        error_msg = str(e)
        print(f"❌ Error generating response: {error_msg}")
        print(f"Traceback: {traceback.format_exc()}")

        # Handle specific error types
        if "413" in error_msg or "too large" in error_msg.lower():
            return "❌ The request is too large. Please try a more specific question or a smaller document."
        elif "rate_limit" in error_msg.lower():
            return "❌ Rate limit exceeded. Please wait a moment and try again."
        elif "expected string or buffer" in error_msg.lower():
            return "❌ There was an issue processing the document content. Please try again with a different question."
        else:
            return generate_fallback_response(documents, query)


def process_user_query(uploaded_file, user_query, memory_manager):
    """Process query against user-uploaded document with error handling"""
    try:
        # Show processing status
        with st.spinner("🔍 Processing your document..."):
            # Create vector store from user upload
            vector_store = process_user_pdf(uploaded_file)

            # Retrieve relevant documents
            retrieved_docs = retrieve_docs(user_query, vector_store)

            # Generate response
            response = answer_query_with_fallback(retrieved_docs, user_query, memory_manager)

        return response

    except Exception as e:
        error_msg = str(e)
        print(f"❌ Error in process_user_query: {error_msg}")
        if "too large" in error_msg.lower():
            return "❌ The document is too large to process. Please try a smaller document or ask more specific questions."
        else:
            return f"❌ Error processing your document: {error_msg}"


def _documents_are_relevant(documents, query):
    """Check if any document is actually relevant to the query"""
    if not documents:
        return False

    query_str = token_manager.safe_string_convert(query)
    query_terms = set(query_str.lower().split())

    for doc in documents:
        content = token_manager.safe_string_convert(doc).lower()
        # Check if any significant terms match
        matching_terms = [term for term in query_terms if len(term) > 3 and term in content]
        if len(matching_terms) >= 1:
            return True
    return False


def should_use_fallback(retrieved_docs: list, rag_response: str) -> bool:
    """
    Determine if we should use fallback response
    """
    if not retrieved_docs:
        return True

    rag_response_str = token_manager.safe_string_convert(rag_response).lower()
    uncertainty_phrases = [
        "don't know", "not in the context", "i don't",
        "no information", "unable to answer", "cannot determine",
        "i couldn't", "isn't covered", "not present in context",
        "based on the provided documents", "the context doesn't"
    ]

    if any(phrase in rag_response_str for phrase in uncertainty_phrases):
        return True

    if len(rag_response_str.split()) < 10:
        return True

    return False


def generate_fallback_response(documents, query):
    """Generate a professional fallback response when AI fails"""
    if documents and _documents_are_relevant(documents, query):
        # We have relevant documents - provide a preview
        best_doc = documents[0]
        preview_content = token_manager.safe_string_convert(best_doc)
        preview = token_manager.truncate_text(preview_content, 200)

        return f"""📜 **Legal Information Found**

{preview}

💡 **This appears relevant to your question, but I'm unable to provide a detailed analysis at the moment.**

🔍 **Suggested**: For detailed legal interpretation, please consult the full document or a qualified legal professional.

*Information sourced from analyzed legal documents*"""

    else:
        # No relevant documents found
        query_str = token_manager.safe_string_convert(query)
        return f"""🔍 **Legal Research Result**

I couldn't find specific information about "{query_str}" in my current legal knowledge base.

**My expertise covers:**
• **Labour Laws**: Employment rights, wages, workplace regulations, disputes
• **Marriage Laws**: Hindu Marriage Act, Special Marriage Act, marriage procedures

**For comprehensive legal guidance:**
1. Consult the specific legal statutes directly
2. Speak with a qualified legal professional  
3. Check official government legal portals

📚 *My knowledge is based on specialized legal documents and may not cover all topics.*"""


def answer_query_with_fallback(documents, query, memory_manager=None):
    """Main function with proper fallback handling and error management"""
    try:
        # First, try to get response from Groq
        rag_response = answer_query(documents, query, memory_manager)

        # Check if we need fallback
        if should_use_fallback(documents, rag_response):
            return generate_fallback_response(documents, query)

        return rag_response

    except Exception as e:
        error_msg = str(e)
        print(f"❌ Unexpected error in answer_query_with_fallback: {error_msg}")
        print(f"Traceback: {traceback.format_exc()}")
        return emergency_response(query)


def get_enhanced_prompt():
    """Enhanced prompt for legal responses with token optimization"""
    template = """
You are an expert AI legal assistant specializing in Indian Labour Laws and Marriage Laws.

CONTEXT FROM LEGAL DOCUMENTS:
{context}

CONVERSATION HISTORY:
{chat_history}

USER QUESTION: {question}

IMPORTANT INSTRUCTIONS:
1. Provide ACCURATE, FACTUAL legal information based ONLY on the context
2. If the context contains relevant information, cite it specifically
3. If the context doesn't contain the answer, say "I don't have specific information on this in my knowledge base"
4. Be clear, practical, and professional
5. Keep your response concise and focused
6. Never make up legal provisions

RESPONSE FORMAT:
- Start with direct answer
- Reference relevant content when possible
- Provide practical implications
- Include important cautions

ANSWER:
"""
    return PromptTemplate(
        template=template,
        input_variables=["chat_history", "context", "question"]
    )


def emergency_response(query):
    """Ultimate fallback for any errors"""
    query_str = token_manager.safe_string_convert(query)
    return f"""⚖️ **Legal Assistant Response**

I'm currently experiencing technical difficulties.

**Your Question:** "{query_str}"

**Suggested Action:** 
- Try asking a more specific question
- Try with a smaller document
- Wait a moment and try again

*My expertise includes Labour Laws and Marriage Laws in India.*"""


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
        print(f"Traceback: {traceback.format_exc()}")


if __name__ == "__main__":
    test_rag_pipeline()