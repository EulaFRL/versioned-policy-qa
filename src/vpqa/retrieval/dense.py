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
        if cache is not None and cache.exists():
            self.emb = np.load(cache)
            assert len(self.emb) == len(self.pids), f"stale embedding cache {cache}"
        else:
            self.emb = self.model.encode([p.text for p in passages], normalize_embeddings=True,
                                         batch_size=32, show_progress_bar=True)
            if cache is not None:
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache, self.emb)

    def search_many(self, queries: list[str], k: int = 20) -> list[list[str]]:
        q = self.model.encode(queries, normalize_embeddings=True, batch_size=64)
        top = np.argsort(-(q @ self.emb.T), axis=1)[:, :k]
        return [[self.pids[i] for i in row] for row in top]

    def search(self, query: str, k: int = 20) -> list[str]:
        return self.search_many([query], k)[0]
