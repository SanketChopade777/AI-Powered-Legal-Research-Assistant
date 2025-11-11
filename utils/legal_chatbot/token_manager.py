# token_manager.py

import tiktoken
from typing import List, Dict, Any
import re


class TokenManager:
    def __init__(self):
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.max_context_tokens = 3000
        self.max_response_tokens = 500

    def count_tokens(self, text: str) -> int:
        """Count tokens in text with proper type checking"""
        if not text or not isinstance(text, str):
            return 0

        try:
            return len(self.encoding.encode(text))
        except Exception as e:
            print(f"❌ Error counting tokens: {e}")
            return len(str(text).split())  # Fallback to word count

    def truncate_text(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token limit with proper error handling"""
        if not text or not isinstance(text, str):
            return ""

        try:
            tokens = self.encoding.encode(text)
            if len(tokens) <= max_tokens:
                return text

            truncated_tokens = tokens[:max_tokens]
            truncated_text = self.encoding.decode(truncated_tokens)
            return truncated_text + "...\n[Content truncated due to size limitations]"

        except Exception as e:
            print(f"❌ Error truncating text: {e}")
            # Simple fallback truncation
            if len(text) > 1000:
                return text[:1000] + "...\n[Content truncated]"
            return text

    def safe_string_convert(self, obj: Any) -> str:
        """Safely convert any object to string"""
        if obj is None:
            return ""
        elif isinstance(obj, str):
            return obj
        elif hasattr(obj, 'page_content'):  # Handle Document objects
            return str(obj.page_content)
        else:
            return str(obj)

    def optimize_context(self, documents: List, query: str) -> str:
        """Optimize context by selecting most relevant parts and managing tokens"""
        if not documents:
            return "No relevant content found."

        # Ensure query is string
        query_str = self.safe_string_convert(query)
        query_lower = query_str.lower()
        query_terms = set(query_lower.split())

        # Remove common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is',
                      'are', 'was', 'were'}
        query_terms = {kw for kw in query_terms if len(kw) > 3 and kw not in stop_words}

        scored_docs = []
        for doc in documents:
            # Safely extract content from document
            content = self.safe_string_convert(doc)
            if not content.strip():
                continue

            content_lower = content.lower()
            score = 0

            # Keyword matching
            for term in query_terms:
                if term in content_lower:
                    score += 3

            # Boost score for legal terms
            legal_terms = ['obligation', 'responsibility', 'payment', 'termination', 'liability',
                           'confidential', 'dispute', 'clause', 'section', 'article', 'party',
                           'contract', 'agreement', 'right', 'duty', 'breach']
            for term in legal_terms:
                if term in query_lower and term in content_lower:
                    score += 2

            if score > 0:
                scored_docs.append((score, content))

        # If no scored docs, use all documents with basic scoring
        if not scored_docs:
            for doc in documents:
                content = self.safe_string_convert(doc)
                if content.strip():
                    scored_docs.append((1, content))  # Basic score

        # Sort by relevance
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        # Build context within token limits
        context_parts = []
        total_tokens = 0

        for score, content in scored_docs[:4]:  # Top 4 most relevant
            if not content.strip():
                continue

            content_tokens = self.count_tokens(content)

            # If adding this content would exceed limit, truncate it
            if total_tokens + content_tokens > self.max_context_tokens:
                remaining_tokens = self.max_context_tokens - total_tokens
                if remaining_tokens > 100:  # Only add if we have meaningful space
                    truncated_content = self.truncate_text(content, remaining_tokens)
                    context_parts.append(truncated_content)
                    total_tokens += self.count_tokens(truncated_content)
                break
            else:
                context_parts.append(content)
                total_tokens += content_tokens

        if not context_parts:
            # Fallback: use beginning of first document
            if documents:
                fallback_content = self.safe_string_convert(documents[0])
                truncated_fallback = self.truncate_text(fallback_content, 1000)
                return f"Relevant content:\n{truncated_fallback}"
            else:
                return "No content available from documents."

        return "\n\n".join(context_parts)