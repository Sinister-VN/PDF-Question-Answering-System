from src.rag import RAGPipeline

PDF_PATH = "data/handbook.pdf"

rag = RAGPipeline(PDF_PATH)

question = "Mục tiêu của tiền xử lý dữ liệu là gì?"

result = rag.ask(question)

print("=" * 80)
print("Question:")
print(result["question"])

print("=" * 80)
print("Answer:")
print(result["answer"])

print("=" * 80)
print("Retrieved Contexts:")

for i, context in enumerate(result["contexts"], start=1):
    print(f"\nContext {i}")
    print("-" * 40)
    print(context)

print("=" * 80)
print("Sources:")

for source in result["sources"]:
    print(source)