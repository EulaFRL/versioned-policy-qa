"""Check that every claim in an answer is supported by a cited passage; otherwise refuse."""
from __future__ import annotations


def verify(answer: str, cited_texts: list[str]) -> tuple[bool, list[str]]:
    raise NotImplementedError
