from typing import List, Optional, Tuple
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from utils.gemini_integration import GeminiIntegration


class QueryRefiner:
    def __init__(self):
        self.llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0.3)

    def refine_query(self, original_query: str, chat_history: List[str] = None) -> str:
        """
        Enhance the original query with context from chat history
        """
        if not chat_history:
            return original_query

        prompt = ChatPromptTemplate.from_template("""
        You are a legal query enhancement system. Improve the clarity and specificity
        of legal questions based on conversation history.

        Chat History:
        {history}

        Current Query: {query}

        Rewrite the current query to be more precise while maintaining legal terminology.

        Enhanced Query:
        """)

        chain = prompt | self.llm
        history_str = "\n".join(chat_history)

        response = chain.invoke({
            "history": history_str,
            "query": original_query
        })

        return response.content.strip()


def should_use_gemini(retrieved_docs: list, rag_response: str) -> Tuple[bool, str]:
    """
    Determine if we should use Gemini as fallback
    """
    if not retrieved_docs:
        return (True, "No documents found")

    rag_response_lower = rag_response.lower()
    trigger_phrases = [
        "don't know", "not in the context", "i don't",
        "no information", "unable to answer", "cannot determine",
        "i couldn't", "isn't covered", "not provided", "not mentioned"
    ]

    if any(phrase in rag_response_lower for phrase in trigger_phrases):
        return (True, "Response indicates missing information")

    if len(rag_response.split()) < 15:  # Slightly longer threshold
        return (True, "Response is too brief")

    return (False, "")