from sentence_transformers import SentenceTransformer
from src.config import EMBEDDING_MODEL

# Load the pre-trained model
embedding_model = SentenceTransformer(EMBEDDING_MODEL)


class EmbeddingModel:
    def __init__(self):

        self.model = SentenceTransformer(
            EMBEDDING_MODEL
        )
    
    def encode(self, texts):

        return self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True
        )