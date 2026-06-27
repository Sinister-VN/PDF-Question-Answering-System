from dotenv import load_dotenv
import os

load_dotenv()

# Embedding Model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Text Splitter
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# Retrieval
TOP_K = 5

# Google Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

GEMINI_MODEL = "gemini-2.5-flash"

TEMPERATURE = 0.3

MAX_OUTPUT_TOKENS = 512