import json
from pathlib import Path
import matplotlib.pyplot as plt

from reliability_lab.ingestion import Chunk
from reliability_lab.retrieval import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    RerankerRetriever,
)
from reliability_lab.evaluation import (
    recall_at_k,
    reciprocal_rank_at_k,
    ndcg_at_k,
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

    k_values = [1, 3, 5, 10]

    all_results = {name: {} for name in retrievers}

    for k in k_values:
        print(f"\n========== TOP {k} ==========")

        for name, retriever in retrievers.items():
            results = []

            for question in questions:
                if not question["answerable"]:
                    continue

                relevant_ids = question["relevant_chunk_ids"]

                if not relevant_ids:
                    raise ValueError(
                        f"No relevant chunks for {question['id']}"
                    )

                retrieved = retriever.search(
                    question["question"], top_k=k
                )
                retrieved_ids = [chunk.id for chunk in retrieved]

                results.append({
                    "recall": recall_at_k(retrieved_ids, relevant_ids, k),
                    "mrr": reciprocal_rank_at_k(retrieved_ids, relevant_ids, k),
                    "ndcg": ndcg_at_k(retrieved_ids, relevant_ids, k),
                })

            averages = {
                metric: sum(r[metric] for r in results) / len(results)
                for metric in ["recall", "mrr", "ndcg"]
            }

            all_results[name][k] = averages

            print(f"\n{name} evaluation ({len(results)} questions)")
            print(f"Recall@{k}: {averages['recall']:.3f}")
            print(f"MRR@{k}:    {averages['mrr']:.3f}")
            print(f"nDCG@{k}:   {averages['ndcg']:.3f}")

    plot_results(all_results, k_values)


def plot_results(all_results, k_values):
    metrics = ["recall", "mrr", "ndcg"]

    for metric in metrics:
        plt.figure(figsize=(8, 5))

        for name, results in all_results.items():
            values = [results[k][metric] for k in k_values]

            plt.plot(
                k_values,
                values,
                marker="o",
                label=name
            )

        plt.title(f"{metric.upper()}@k comparison")
        plt.xlabel("Top-k")
        plt.ylabel(metric.upper())
        plt.xticks(k_values)
        plt.ylim(0, 1)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        from pathlib import Path

        output_dir = Path("results/figures")
        output_dir.mkdir(parents=True, exist_ok=True)

        plt.savefig(output_dir / f"{metric}_comparison.png", dpi=300)
        plt.show()


if __name__ == "__main__":
    main()
