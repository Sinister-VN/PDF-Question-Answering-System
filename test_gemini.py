from src.gemini_client import GeminiClient
from src.config import GEMINI_API_KEY

client = GeminiClient()

response = client.generate(
    "Explain what Retrieval-Augmented Generation is in one sentence."
)

print(response)