import math


def recall_at_k(retrieved_ids, relevant_ids, k):
    relevant = set(relevant_ids)
    retrieved = set(retrieved_ids[:k])

    if not relevant:
        return 0.0

    return len(retrieved & relevant) / len(relevant)


def reciprocal_rank_at_k(retrieved_ids, relevant_ids, k):
    relevant = set(relevant_ids)

    for rank, chunk_id in enumerate(retrieved_ids[:k], start=1):
        if chunk_id in relevant:
            return 1 / rank

    return 0.0


def ndcg_at_k(retrieved_ids, relevant_ids, k):
    relevant = set(relevant_ids)

    if not relevant:
        return 0.0

    dcg = sum(
        1 / math.log2(rank + 1)
        for rank, chunk_id in enumerate(retrieved_ids[:k], start=1)
        if chunk_id in relevant
    )

    idcg = sum(
        1 / math.log2(rank + 1)
        for rank in range(1, min(len(relevant), k) + 1)
    )

    return dcg / idcg
