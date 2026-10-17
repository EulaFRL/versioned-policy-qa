from __future__ import annotations

import numpy as np

from vpqa.schema import Passage


class DenseIndex:
    def __init__(self, passages: list[Passage], model_name: str = "BAAI/bge-m3"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.pids = [p.pid for p in passages]
        self.emb = self.model.encode([p.text for p in passages], normalize_embeddings=True)

    def search(self, query: str, k: int = 20) -> list[str]:
        q = self.model.encode([query], normalize_embeddings=True)[0]
        order = np.argsort(-self.emb @ q)[:k]
        return [self.pids[i] for i in order]
