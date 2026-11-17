
import json
from pathlib import Path

from reliability_lab.ingestion import Chunk
from reliability_lab.retrieval import BM25Retriever
from reliability_lab.generation import RAGGenerator


def main():
    # Load chunks
    chunks_path = Path("data/benchmark/chunks.json")

    with chunks_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    chunks = [Chunk(**item) for item in data]

    # Initialize retriever
    retriever = BM25Retriever(chunks)

    # Initialize LLM
    generator = RAGGenerator()

    # Ask a question
    question = "What is the main advantage of the Transformer architecture?"

    # Retrieve relevant chunks
    retrieved_chunks = retriever.search(question, top_k=5)

    # Generate answer
    answer = generator.generate(question, retrieved_chunks)

    print(f"Question: {question}")
    print(f"\nAnswer: {answer}")


if __name__ == "__main__":
    main()
