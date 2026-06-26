from sentence_transformers import SentenceTransformer
from src.config import EMDBEDDING_MODEL

# Load the pre-trained model
embedding_model = SentenceTransformer(EMDBEDDING_MODEL)


class EmbeddingModel:
    def __init__(self):

        self.model = SentenceTransformer(
            EMDBEDDING_MODEL
        )
    
    def encode(self, texts):

        return self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True
        )