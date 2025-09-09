from gemini_integration import GeminiIntegration


def test_gemini():
    gemini = GeminiIntegration()

    print("\n=== Testing API Availability ===")
    print("API Available:", gemini.is_available())

    print("\n=== Testing Legal Query ===")
    legal_response = gemini.generate_response("Explain Article 21 of the Indian Constitution")
    print("Response:", legal_response)

    print("\n=== Testing Non-Legal Query ===")
    general_response = gemini.generate_response("What is photosynthesis?")
    print("Response:", general_response)


if __name__ == "__main__":
    test_gemini()