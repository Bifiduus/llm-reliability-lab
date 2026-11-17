import json
from pathlib import Path

from reliability_lab.ingestion import Chunk
from reliability_lab.generation import RAGGenerator


def main():
    with Path("data/benchmark/chunks.json").open(encoding="utf-8") as f:
        chunks = [Chunk(**data) for data in json.load(f)]

    chunks_by_id = {chunk.id: chunk for chunk in chunks}

    with Path("results/retrieved_contexts.json").open(encoding="utf-8") as f:
        contexts = json.load(f)

    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "rag_results.json"

    if output_path.exists():
        with output_path.open(encoding="utf-8") as f:
            results = json.load(f)
    else:
        results = []

    completed = {
        (result["retriever"], result["question_id"], result["top_k"])
        for result in results
    }

    generator = RAGGenerator()

    for context in contexts:
        key = (
            context["retriever"],
            context["question_id"],
            context["top_k"],
        )

        if key in completed:
            print(
                f"Skipping {context['retriever']} / {context['question_id']}")
            continue

        retrieved = [
            chunks_by_id[chunk_id]
            for chunk_id in context["retrieved_chunk_ids"]
        ]

        answer = generator.generate(
            context["question"],
            retrieved,
        )

        result = {
            **context,
            "generated_answer": answer,
        }

        results.append(result)

        temporary_path = output_path.with_suffix(".tmp")

        with temporary_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        temporary_path.replace(output_path)
        completed.add(key)

        print(f"\nQuestion: {context['question']}")
        print(f"Answer: {answer}")

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
