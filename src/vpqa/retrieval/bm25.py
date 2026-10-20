from __future__ import annotations

import re

import numpy as np
from rank_bm25 import BM25Okapi

from vpqa.schema import Passage

_TOKEN = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25Index:
    def __init__(self, passages: list[Passage]):
        self.pids = [p.pid for p in passages]
        self.domains = np.array([p.domain for p in passages])
        self.bm25 = BM25Okapi([tokenize(p.text) for p in passages])

    def search(self, query: str, k: int = 20, domain: str | None = None) -> list[str]:
        """Top k passages, restricted to one domain when `domain` is given."""
        scores = self.bm25.get_scores(tokenize(query))
        if domain is not None:
            scores = np.where(self.domains == domain, scores, -np.inf)
        order = np.argsort(-scores, kind="stable")[:k]
        return [self.pids[i] for i in order]
