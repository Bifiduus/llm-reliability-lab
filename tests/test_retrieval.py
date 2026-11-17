from reliability_lab.ingestion import Chunk
from reliability_lab.retrieval import BM25Retriever
from reliability_lab.retrieval import RerankerRetriever

import numpy as np
from unittest.mock import Mock


def test_bm25_retrieval():
    chunks = [
        Chunk(
            id="1",
            document="test.pdf",
            page=1,
            text="The transformer uses self attention mechanisms",
        ),
        Chunk(
            id="2",
            document="test.pdf",
            page=2,
            text="Monte Carlo methods estimate expected rewards",
        ),
        Chunk(
            id="3",
            document="test.pdf",
            page=3,
            text="Neural networks learn representations",
        ),
    ]

    retriever = BM25Retriever(chunks)

    results = retriever.search("Monte Carlo rewards", top_k=1)

    assert len(results) == 1
    assert results[0].id == "2"


def test_reranker_reorders_chunks():
    chunks = [
        Chunk("1", "test.pdf", 1, "First passage"),
        Chunk("2", "test.pdf", 1, "Second passage"),
        Chunk("3", "test.pdf", 1, "Third passage"),
    ]

    retriever = Mock()
    retriever.search.return_value = chunks

    reranker = RerankerRetriever.__new__(RerankerRetriever)
    reranker.retriever = retriever
    reranker.candidate_k = 3
    reranker.model = Mock()
    reranker.model.predict.return_value = np.array([0.2, 0.9, 0.5])

    results = reranker.search("test question", top_k=2)

    assert [chunk.id for chunk in results] == ["2", "3"]
