import streamlit as st
from groq import Groq
from typing import Dict
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()


class DocumentRAGAnalyzer:
    def __init__(self):
        self.groq_client = self._initialize_groq_client()

    def _initialize_groq_client(self):
        """Initialize Groq client with the working model"""
        try:
            groq_api_key = os.getenv("GROQ_API_KEY")
            if not groq_api_key:
                st.error("❌ GROQ_API_KEY not found in .env file")
                return None

            # Use the same model that works in your legal chatbot
            client = Groq(api_key=groq_api_key)

            # Test the connection
            test_response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5
            )
            print("✅ Groq client initialized successfully with llama-3.1-8b-instant")
            return client

        except Exception as e:
            st.error(f"❌ Error initializing Groq client: {e}")
            return None

    def is_configured(self):
        return self.groq_client is not None

    def generate_summary(self, text: str, document_name: str) -> Dict:
        """Generate comprehensive summary using Groq"""
        if not self.is_configured():
            return {"error": "Groq API not configured"}

        try:
            prompt = f"""
            Analyze this legal document and provide a comprehensive yet concise summary.

            DOCUMENT NAME: {document_name}
            DOCUMENT CONTENT:
            {text[:8000]}  # Limit context length for faster processing

            Please provide a structured summary covering:

            1. **Document Type & Purpose**: What type of legal document is this and what is its main purpose?
            2. **Key Parties**: Who are the main parties involved?
            3. **Main Obligations**: What are the key responsibilities and obligations?
            4. **Important Terms**: What are the critical terms, dates, and conditions?
            5. **Key Clauses**: What are the most important clauses or sections?
            6. **Overall Assessment**: Brief assessment of the document's completeness and clarity.

            Keep the summary clear, professional, and easy to understand. Focus on the most important aspects.
            """

            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",  # Use the working model
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert legal document analyst. Provide clear, accurate summaries 
                        that help users quickly understand legal documents. Be specific and focus on practical insights."""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=1200  # Reduced for faster response
            )

            return {
                "analysis_type": "Summary",
                "content": response.choices[0].message.content,
                "timestamp": self._get_current_timestamp()
            }

        except Exception as e:
            return {"error": f"Summary generation failed: {str(e)}"}

    def answer_question(self, text: str, question: str, document_name: str = "uploaded_document") -> str:
        """Answer specific questions about the document"""
        if not self.is_configured():
            return "❌ Groq API not configured"

        try:
            # Create context from relevant parts of the document
            context = self._find_relevant_context(text, question)

            prompt = f"""
            Based EXCLUSIVELY on the following legal document content, answer the user's question.

            DOCUMENT: {document_name}
            RELEVANT DOCUMENT CONTENT:
            {context}

            USER QUESTION: {question}

            IMPORTANT INSTRUCTIONS:
            1. Answer based ONLY on the provided document content
            2. If the information is not in the document, clearly state: "This specific information is not found in the provided document."
            3. Be precise and reference specific content when possible
            4. If you're unsure, indicate that the information might not be in the document
            5. Do not make up or assume any information not present in the document

            ANSWER:
            """

            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",  # Use the working model
                messages=[
                    {
                        "role": "system",
                        "content": """You are a precise legal assistant that answers questions based ONLY on provided legal documents. 
                        Never use external knowledge. Be accurate and honest about what information is available."""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=800  # Reduced for faster response
            )

            return response.choices[0].message.content

        except Exception as e:
            return f"❌ Error answering question: {str(e)}"

    def _find_relevant_context(self, text: str, question: str) -> str:
        """Find relevant parts of the document for the question"""
        # Simple keyword-based context extraction
        question_keywords = set(question.lower().split())
        paragraphs = [p for p in text.split('\n\n') if p.strip()]

        relevant_paragraphs = []

        for para in paragraphs:
            para_lower = para.lower()
            # Check if paragraph contains any question keywords
            matching_keywords = [kw for kw in question_keywords if len(kw) > 3 and kw in para_lower]
            if len(matching_keywords) >= 1:
                relevant_paragraphs.append(para)

            # Limit to top 3 most relevant paragraphs for faster processing
            if len(relevant_paragraphs) >= 3:
                break

        if not relevant_paragraphs:
            # If no specific matches, return the beginning of the document
            return "\n".join(paragraphs[:2])

        return "\n\n".join(relevant_paragraphs)

    def analyze_document_comprehensive(self, text: str, analysis_type: str, document_name: str) -> Dict:
        """Perform comprehensive analysis of the document"""
        if not self.is_configured():
            return {"error": "Groq API not configured"}

        try:
            if analysis_type == "risk":
                prompt = self._create_risk_analysis_prompt(text, document_name)
            elif analysis_type == "clauses":
                prompt = self._create_clause_analysis_prompt(text, document_name)
            elif analysis_type == "compliance":
                prompt = self._create_compliance_analysis_prompt(text, document_name)
            else:
                prompt = self._create_full_analysis_prompt(text, document_name)

            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",  # Use the working model
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert legal document analyst. Provide detailed, accurate analysis 
                        based on the document content. Be specific and practical in your recommendations."""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=1500  # Reduced for faster processing
            )

            return {
                "analysis_type": analysis_type.capitalize(),
                "content": response.choices[0].message.content,
                "timestamp": self._get_current_timestamp()
            }

        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}

    def _create_risk_analysis_prompt(self, text: str, document_name: str) -> str:
        return f"""
        Conduct a risk assessment of this legal document.

        DOCUMENT: {document_name}
        CONTENT:
        {text[:6000]}

        Identify:
        1. **High-Risk Areas**: Clauses that pose significant risks
        2. **Potential Issues**: Ambiguous language or missing protections
        3. **Recommendations**: Actionable suggestions to mitigate risks

        Be specific and practical.
        """

    def _create_clause_analysis_prompt(self, text: str, document_name: str) -> str:
        return f"""
        Extract and analyze key clauses from this legal document.

        DOCUMENT: {document_name}
        CONTENT:
        {text[:6000]}

        For each significant clause, provide:
        - **Clause Type**
        - **Key Provisions** 
        - **Potential Implications**
        - **Recommendations**

        Focus on the most important clauses.
        """

    def _create_compliance_analysis_prompt(self, text: str, document_name: str) -> str:
        return f"""
        Evaluate this legal document for compliance.

        DOCUMENT: {document_name}
        CONTENT:
        {text[:6000]}

        Assess:
        1. **Regulatory Compliance**: Potential issues
        2. **Standard Practices**: Adherence to conventions
        3. **Improvement Opportunities**: Specific suggestions

        Focus on practical compliance issues.
        """

    def _create_full_analysis_prompt(self, text: str, document_name: str) -> str:
        return f"""
        Provide a comprehensive analysis of this legal document.

        DOCUMENT: {document_name}
        CONTENT:
        {text[:6000]}

        Cover:
        - Document type and purpose
        - Key terms and conditions  
        - Risk assessment
        - Key clauses
        - Overall recommendations

        Be thorough but concise.
        """

    def _get_current_timestamp(self):
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")