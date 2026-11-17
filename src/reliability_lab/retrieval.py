from rank_bm25 import BM25Okapi
import re
from reliability_lab.ingestion import Chunk
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Retriever:

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks

        self.tokenized_corpus = [
            tokenize(chunk.text)
            for chunk in chunks
        ]

        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query: str, top_k: int = 5) -> list[Chunk]:
        tokenized_query = tokenize(query)

        scores = self.bm25.get_scores(tokenized_query)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )

        return [
            self.chunks[i]
            for i in ranked_indices[:top_k]
        ]


class DenseRetriever:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        self.embeddings = self.model.encode(
            [chunk.text for chunk in chunks],
            normalize_embeddings=True,
            convert_to_numpy=True
        )

    def search(self, query: str, top_k: int = 5) -> list[Chunk]:
        query_embedding = self.model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True
        )

        scores = self.embeddings @ query_embedding

        ranked_indices = np.argsort(scores)[::-1][:top_k]

        return [self.chunks[i] for i in ranked_indices]


class HybridRetriever:
    def __init__(
        self,
        bm25: BM25Retriever,
        dense: DenseRetriever,
        rrf_k: int = 60,
        candidate_k: int = 50
    ):
        self.bm25 = bm25
        self.dense = dense
        self.rrf_k = rrf_k
        self.candidate_k = candidate_k

    def search(self, query: str, top_k: int = 5) -> list[Chunk]:
        bm25_results = self.bm25.search(query, top_k=self.candidate_k)
        dense_results = self.dense.search(query, top_k=self.candidate_k)

        scores = {}
        chunks_by_id = {}

        for results in (bm25_results, dense_results):
            for rank, chunk in enumerate(results, start=1):
                scores[chunk.id] = (
                    scores.get(chunk.id, 0)
                    + 1 / (self.rrf_k + rank)
                )
                chunks_by_id[chunk.id] = chunk

        ranked_ids = sorted(
            scores,
            key=lambda chunk_id: scores[chunk_id],
            reverse=True
        )

        return [chunks_by_id[chunk_id] for chunk_id in ranked_ids[:top_k]]


class RerankerRetriever:
    def __init__(
        self,
        retriever: HybridRetriever,
        candidate_k: int = 20
    ):
        self.retriever = retriever
        self.candidate_k = candidate_k

        self.model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

    def search(self, query: str, top_k: int = 5) -> list[Chunk]:
        candidates = self.retriever.search(
            query,
            top_k=max(self.candidate_k, top_k)
        )

        if not candidates:
            return []

        pairs = [
            (query, chunk.text)
            for chunk in candidates
        ]

        scores = self.model.predict(pairs)

        ranked_indices = np.argsort(scores)[::-1]

        return [
            candidates[i]
            for i in ranked_indices[:top_k]
        ]
