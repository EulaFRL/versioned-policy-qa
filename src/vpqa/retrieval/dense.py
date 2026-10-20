from __future__ import annotations

from pathlib import Path

import numpy as np

from vpqa.schema import Passage


class DenseIndex:
    def __init__(self, passages: list[Passage], model_name: str = "BAAI/bge-m3",
                 cache: Path | None = None, max_seq_length: int = 512):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.model.max_seq_length = max_seq_length
        self.pids = [p.pid for p in passages]
        self.domains = np.array([p.domain for p in passages])
        if cache is not None and cache.exists():
            self.emb = np.load(cache)
            assert len(self.emb) == len(self.pids), f"stale embedding cache {cache}"
        else:
            self.emb = self.model.encode([p.text for p in passages], normalize_embeddings=True,
                                         batch_size=32, show_progress_bar=True)
            if cache is not None:
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache, self.emb)

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True, batch_size=64)

    def search_many(self, queries: list[str], k: int = 20,
                    domains: list[str | None] | None = None) -> list[list[str]]:
        """Top k passages per query; `domains[i]` restricts query i to one domain (None: no filter)."""
        scores = self.encode(queries) @ self.emb.T
        if domains is not None:
            for i, d in enumerate(domains):
                if d is not None:
                    scores[i, self.domains != d] = -np.inf
        top = np.argsort(-scores, axis=1, kind="stable")[:, :k]
        return [[self.pids[i] for i in row] for row in top]

    def search(self, query: str, k: int = 20, domain: str | None = None) -> list[str]:
        return self.search_many([query], k, [domain])[0]
