import os
import json
import requests
from typing import Optional
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential
import time


class GeminiIntegration:
    def __init__(self):
        self.api_key = self._load_api_key()
        if self.api_key:
            genai.configure(api_key=self.api_key)
        else:
            print("❌ ERROR: No API key found")

    def _load_api_key(self) -> Optional[str]:
        """Load API key from config.json"""
        try:
            with open("config.json", "r") as f:
                config = json.load(f)
            return config.get("GOOGLE_API_KEY")
        except Exception as e:
            print(f"Error loading API key: {e}")
            return None

    def is_available(self) -> bool:
        """Check API availability"""
        if not self.api_key:
            return False
        try:
            models = genai.list_models()
            return any(model.name.startswith('models/') for model in models)
        except Exception as e:
            print(f"API check failed: {e}")
            return False

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate_response(self, query: str, context: str = None) -> str:
        """Generate response using free model"""
        if not self.api_key:
            return "🔒 API key missing"

        try:
            prompt = self._build_prompt(query, context)
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)
            return self._format_legal_response(response.text)
        except Exception as e:
            print(f"Generation Error: {e}")
            return f"⚠️ Error: {str(e)}"

    def _format_legal_response(self, text: str) -> str:
        """Format legal response with proper headings and structure"""
        if not text:
            return "No response generated"

        sections = text.split('\n\n')
        formatted = []
        for section in sections:
            if section.strip().endswith(':'):  # Likely a heading
                formatted.append(f"**{section}**")
            elif any(char.isdigit() for char in section[:20]):  # Contains legal references
                formatted.append(f"📜 {section}")
            else:
                formatted.append(section)
        return '\n\n'.join(formatted)

    def _build_prompt(self, query: str, context: str = None) -> str:
        """Build the prompt for Gemini with context"""
        base_prompt = """You are an AI legal assistant. Provide accurate, professional legal information.
        For general legal questions, include:
        1) Definition
        2) Key legal principles
        3) Typical process
        4) Important considerations

        Important instructions:
        1. Keep responses concise but comprehensive
        2. Always clarify when laws may vary by jurisdiction
        3. For non-legal questions, say "This appears to be a non-legal question"
        """

        if context:
            return f"""{base_prompt}
            RELEVANT CONTEXT:
            {context}

            QUESTION:
            {query}
            """
        return f"""{base_prompt}
        QUESTION:
        {query}
        """