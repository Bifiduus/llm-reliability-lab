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

    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "rag_results.json"

    # Load previously saved results, if any.
    if output_path.exists():
        with output_path.open(encoding="utf-8") as f:
            results = json.load(f)
    else:
        results = []

    completed = {
        (result["retriever"], result["question_id"], result["top_k"])
        for result in results
    }

    for name, retriever in retrievers.items():
        print(f"\n========== {name} ==========")

        for question in questions:
            key = (name, question["id"], top_k)

            if key in completed:
                print(f"Skipping {name} / {question['id']}")
                continue

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

            # Save after every successful generation.
            temporary_path = output_path.with_suffix(".tmp")

            with temporary_path.open("w", encoding="utf-8") as f:
                json.dump(
                    results,
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

            temporary_path.replace(output_path)
            completed.add(key)

            print(f"\nQuestion: {question['question']}")
            print(f"Answer: {answer}")

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
