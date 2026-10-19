from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from vpqa.schema import Passage

_TOKEN = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25Index:
    def __init__(self, passages: list[Passage]):
        self.pids = [p.pid for p in passages]
        self.bm25 = BM25Okapi([tokenize(p.text) for p in passages])

    def search(self, query: str, k: int = 20) -> list[str]:
        scores = self.bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [self.pids[i] for i in order]
