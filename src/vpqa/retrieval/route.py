"""Department routing: predict the domain of a turn and restrict retrieval to it when confident.

A MultiDoc2Dial dialogue stays inside one domain, so unlike the retrieval query (vpqa.retrieval.query)
the routing text can use the whole dialogue so far. The classifier is logistic regression on bge-m3
embeddings, trained on the train split (scripts/train_router.py).
"""
from __future__ import annotations

import numpy as np

from vpqa.schema import Turn


def routing_text(turn: Turn, mode: str = "dialogue", max_words: int = 256) -> str:
    """`current`: the current utterance. `dialogue`: current utterance, then the history most recent
    first, truncated to `max_words` so the oldest utterances are dropped first."""
    if mode == "current":
        return turn.query
    if mode != "dialogue":
        raise ValueError(f"unknown routing mode: {mode}")
    return " ".join(" ".join([turn.query, *reversed(turn.history)]).split()[:max_words])


class DomainRouter:
    def __init__(self, c: float = 1.0):
        from sklearn.linear_model import LogisticRegression

        self.clf = LogisticRegression(C=c, max_iter=2000)

    def fit(self, emb: np.ndarray, domains: list[str]) -> DomainRouter:
        self.clf.fit(emb, domains)
        return self

    def predict(self, emb: np.ndarray) -> tuple[list[str], np.ndarray]:
        """Most likely domain and its probability for each row."""
        proba = self.clf.predict_proba(emb)
        return [self.clf.classes_[i] for i in proba.argmax(1)], proba.max(1)

    def route(self, emb: np.ndarray, threshold: float | list[float]) -> list[str | None]:
        """Domain to restrict retrieval to, or None (search everything) below the threshold, which can
        be one value or one per row."""
        domains, conf = self.predict(emb)
        thresholds = np.broadcast_to(threshold, conf.shape)
        return [d if c >= t else None for d, c, t in zip(domains, conf, thresholds)]


def thresholds_for(turns: list[Turn], first_turn: float, follow_up: float) -> list[float]:
    """Per-turn confidence threshold: a first turn has only one utterance to route on."""
    return [follow_up if t.history else first_turn for t in turns]
