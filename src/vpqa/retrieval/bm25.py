from __future__ import annotations

from rank_bm25 import BM25Okapi

from vpqa.schema import Passage


class BM25Index:
    def __init__(self, passages: list[Passage]):
        self.pids = [p.pid for p in passages]
        self.bm25 = BM25Okapi([p.text.lower().split() for p in passages])

    def search(self, query: str, k: int = 20) -> list[str]:
        scores = self.bm25.get_scores(query.lower().split())
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [self.pids[i] for i in order]
