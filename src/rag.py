from src.loader import load_pdf
from src.splitter import split_documents
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore
from src.gemini_client import GeminiClient
from src.prompt import PromptTemplate

class RAGPipeline:

    def __init__(self, pdf_path):

        # 1. Load PDF
        documents = load_pdf(pdf_path)

        # 2. Split
        self.chunks = split_documents(documents)

        # 3. Extract texts
        texts = [chunk.page_content for chunk in self.chunks]

        # 4. Create EmbeddingModel
        self.embedding_model = EmbeddingModel()

        # 5. Encode texts
        embeddings = self.embedding_model.encode(texts)

        # 6. Build VectorStore
        self.vector_store = VectorStore(
            embeddings
        )

        # 7. Create PromptTemplate
        self.prompt_builder = PromptTemplate()

        # 8. Create GeminiClient
        self.gemini_client = GeminiClient()

    def ask(self, question):

    # 1. Encode question
        query_embedding = self.embedding_model.encode([question])

    # 2. Search
        distances, indices = self.vector_store.search(
            query_embedding
        )
    # 3. Get chunks
        contexts = [
            self.chunks[idx].page_content 
            for idx in indices[0]
        ]
    # 4. Metadata for contexts
        sources = [
            self.chunks[idx].metadata
            for idx in indices[0]
        ]

    # 5. Build prompt
        prompt = self.prompt_builder.build(
            contexts=contexts,
            question=question
        )

    # 6. Generate answer
        answer = self.gemini_client.generate(
            prompt
        )

    # 7. Return result
        return {
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "sources": sources,
            "indices": indices.tolist(),
            "distances": distances.tolist()
        }