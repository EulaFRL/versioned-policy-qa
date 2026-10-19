"""Build the retrieval query for a turn from the current utterance and dialogue history."""
from __future__ import annotations

from vpqa.schema import Turn


def build_query(turn: Turn, mode: str = "last", max_words: int = 128) -> str:
    """`last`: current user utterance only. `history`: current utterance followed by earlier turns,
    most recent first, truncated to `max_words` (the MultiDoc2Dial retriever convention)."""
    if mode == "last":
        return turn.query
    if mode == "history":
        words = " ".join([turn.query, *reversed(turn.history)]).split()
        return " ".join(words[:max_words])
    raise ValueError(f"unknown query mode: {mode}")
