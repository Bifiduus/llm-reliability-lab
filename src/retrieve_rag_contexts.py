import json
from pathlib import Path

from reliability_lab.ingestion import Chunk
from reliability_lab.retrieval import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    RerankerRetriever,
)


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

    top_k = 5
    results = []

    for name, retriever in retrievers.items():
        print(f"\n========== {name} ==========")

        for question in questions:
            retrieved = retriever.search(
                question["question"],
                top_k=top_k,
            )

            results.append({
                "question_id": question["id"],
                "question": question["question"],
                "reference_answer": question["answer"],
                "answerable": question["answerable"],
                "retriever": name,
                "top_k": top_k,
                "retrieved_chunk_ids": [
                    chunk.id for chunk in retrieved
                ],
            })

    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "retrieved_contexts.json"

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nContexts saved to {output_path}")


if __name__ == "__main__":
    main()
