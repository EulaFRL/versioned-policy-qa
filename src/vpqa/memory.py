"""Memory in three layers, from least to most authoritative:

- session: what this conversation is about, kept as the documents each of the system's replies cited
  (Session). It is a hint for retrieval, never evidence: the next turn still cites passages it retrieved.
- organisational knowledge: answers that passed the verifier, stored with the exact passage versions they
  cite (AnswerMemory). A stored answer is reused only while every cited version is still in force on the
  new question's date; otherwise it is dropped and the question goes through the pipeline again.
- facts: the documents themselves, the only thing an answer may cite.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Callable

import numpy as np

from vpqa.pipeline.verifier import in_force
from vpqa.schema import Passage


class Session:
    """The documents cited by each reply of one conversation, most recent reply first."""

    def __init__(self):
        self.replies: list[list[str]] = []

    def record(self, doc_ids: list[str]) -> None:
        self.replies.insert(0, list(dict.fromkeys(doc_ids)))

    def focus(self, replies: int = 1) -> list[str]:
        """Documents cited by the last `replies` replies, most recent first."""
        return list(dict.fromkeys(d for r in self.replies[:replies] for d in r))


@dataclass
class Entry:
    question: str
    vector: np.ndarray
    answer: str
    cited: list[str]  # passage versions (pids) the answer cites


class AnswerMemory:
    """Verified answers, looked up by question similarity (bge-m3 cosine of normalised vectors).

    With `bind`, an entry is a candidate only if every passage version it cites is in force on the new
    question's date; among the candidates at or above `threshold`, the most similar is reused. Without
    `bind`, the most similar entry is reused whatever its versions: the cache a plain FAQ store would be.
    """

    def __init__(self, encode: Callable[[list[str]], np.ndarray], passages: dict[str, Passage],
                 threshold: float = 0.9, bind: bool = True):
        self.encode, self.passages, self.threshold, self.bind = encode, passages, threshold, bind
        self.entries: list[Entry] = []

    def store(self, question: str, answer: str, cited: list[str], vector: np.ndarray | None = None) -> bool:
        """Keep a verified answer; one that cites nothing (a clarifying question) has no versions to bind
        to and is not kept."""
        if not cited:
            return False
        if vector is None:
            vector = self.encode([question])[0]
        self.entries.append(Entry(question, vector, answer, list(cited)))
        return True

    def valid(self, entry: Entry, on: date | None) -> bool:
        return all(in_force(self.passages[p], on) for p in entry.cited)

    def lookup(self, question: str, on: date | None, vector: np.ndarray | None = None) -> tuple[Entry | None, str]:
        """The entry to reuse and why: "hit", "miss" (nothing similar enough) or "stale" (something similar
        enough, but every such entry cites a version not in force on `on`)."""
        if not self.entries:
            return None, "miss"
        if vector is None:
            vector = self.encode([question])[0]
        sims = np.stack([e.vector for e in self.entries]) @ vector
        close = [i for i in np.argsort(-sims) if sims[i] >= self.threshold]
        if not close:
            return None, "miss"
        if self.bind:
            close = [i for i in close if self.valid(self.entries[i], on)]
            if not close:
                return None, "stale"
        return self.entries[close[0]], "hit"
