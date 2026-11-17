
import json
from pathlib import Path

from reliability_lab.ingestion import Chunk
from reliability_lab.retrieval import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    RerankerRetriever,
)
from reliability_lab.generation import RAGGenerator


def main():
    with Path("data/benchmark/chunks.json").open(encoding="utf-8") as f:
        chunks = [Chunk(**data) for data in json.load(f)]

    with Path("data/benchmark/questions.json").open(encoding="utf-8") as f:
        questions = json.load(f)

    bm25 = BM25Retriever(chunks)
    dense = DenseRetriever(chunks)
    hybrid = HybridRetriever(bm25, dense)
    reranker = RerankerRetriever(hybrid)

    retrievers = {
        "BM25": bm25,
        "Dense": dense,
        "Hybrid": hybrid,
        "Reranker": reranker,
    }

    generator = RAGGenerator()
    top_k = 5

    results = []

    for name, retriever in retrievers.items():
        print(f"\n========== {name} ==========")

        for question in questions:
            retrieved = retriever.search(
                question["question"],
                top_k=top_k,
            )

            answer = generator.generate(
                question["question"],
                retrieved,
            )

            result = {
                "question_id": question["id"],
                "question": question["question"],
                "reference_answer": question["answer"],
                "answerable": question["answerable"],
                "retriever": name,
                "top_k": top_k,
                "retrieved_chunk_ids": [
                    chunk.id for chunk in retrieved
                ],
                "generated_answer": answer,
            }

            results.append(result)

            print(f"\nQuestion: {question['question']}")
            print(f"Answer: {answer}")

    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "rag_results.json"

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
