import pytest

from reliability_lab.evaluation import (
    recall_at_k,
    reciprocal_rank_at_k,
    ndcg_at_k,
)


def test_recall_at_k():
    retrieved = ["a", "b", "c"]
    relevant = ["b", "c"]

    assert recall_at_k(retrieved, relevant, 1) == 0
    assert recall_at_k(retrieved, relevant, 2) == 0.5
    assert recall_at_k(retrieved, relevant, 3) == 1


def test_mrr_at_k():
    retrieved = ["a", "b", "c"]
    relevant = ["b"]

    assert reciprocal_rank_at_k(retrieved, relevant, 1) == 0
    assert reciprocal_rank_at_k(retrieved, relevant, 3) == 0.5


def test_ndcg_at_k():
    retrieved = ["a", "b", "c"]
    relevant = ["b", "c"]

    assert ndcg_at_k(retrieved, relevant, 1) == 0
    assert ndcg_at_k(retrieved, relevant, 3) == pytest.approx(
        0.6934, abs=0.001
    )


def test_perfect_ranking():
    retrieved = ["a", "b"]
    relevant = ["a", "b"]

    assert recall_at_k(retrieved, relevant, 2) == 1
    assert reciprocal_rank_at_k(retrieved, relevant, 2) == 1
    assert ndcg_at_k(retrieved, relevant, 2) == 1


def test_no_relevant_chunks():
    retrieved = ["a", "b"]

    assert recall_at_k(retrieved, [], 2) == 0
    assert reciprocal_rank_at_k(retrieved, [], 2) == 0
    assert ndcg_at_k(retrieved, [], 2) == 0
