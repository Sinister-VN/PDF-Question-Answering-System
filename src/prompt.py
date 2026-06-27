class PromptTemplate:
    def __init__(self):
        self.template = """
You are an AI assistant specialized in answering questions based on the provided document.

Instructions:
- Answer ONLY using the provided context.
- Do not use your own knowledge.
- If the answer cannot be found in the context, reply:
   "I don't have enough information from the provided document."
- Keep your answer clear and concise.

Context:
{context}

Question: 
{question}

Answer:
"""
    def build(
            self, 
            contexts: list[str], 
            question: str
    ) -> str:
        
        context = "\n\n".join(contexts)

        return self.template.format(
            context=context,
            question=question
        )