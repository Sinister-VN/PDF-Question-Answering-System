import faiss
import numpy as np

class VectorStore:
    def __init__(self, embeddings):
        self.dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatL2(self.dimension)

        self.index.add(
            embeddings.astype(np.float32)
        )
    def search(self, query_embedding, k=5):
        
        distances, indices = self.index.search(
            query_embedding.astype(np.float32),
            k
        )

        return distances, indices