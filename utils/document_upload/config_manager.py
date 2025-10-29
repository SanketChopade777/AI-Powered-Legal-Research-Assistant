import os
from dotenv import load_dotenv


class ConfigManager:
    def __init__(self):
        load_dotenv()  # Load environment variables from .env file
        self._validate_config()

    def _validate_config(self):
        """Validate that required configuration is present"""
        groq_key = self.get_groq_api_key()
        if not groq_key:
            print("⚠️ GROQ_API_KEY not found in .env file")

    def get_groq_api_key(self) -> str:
        """Get Groq API key from environment variables"""
        return os.getenv("GROQ_API_KEY")
