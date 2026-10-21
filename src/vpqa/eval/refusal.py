"""Unanswerable variants of answerable turns, and the scores of answering or declining.

A turn becomes unanswerable when every passage of the documents holding its gold passages is withheld
from retrieval (Turn.withheld). The pair (original, withheld) shares the query and history, so the only
difference the system can react to is whether the evidence is there. Other documents may still answer a
withheld turn; such turns count as errors when answered, so refusal precision is a lower bound.
"""
from __future__ import annotations

from dataclasses import replace

from vpqa.schema import Passage, Turn


def withhold_gold_docs(turns: list[Turn], by_pid: dict[str, Passage]) -> list[Turn]:
    by_doc: dict[str, list[str]] = {}
    for p in by_pid.values():
        by_doc.setdefault(p.doc_id, []).append(p.pid)
    gold_docs = lambda t: {by_pid[g].doc_id for g in t.gold_pids}
    return [replace(t, answerable=False, withheld=[p for d in gold_docs(t) for p in by_doc[d]]) for t in turns]


def best_threshold(scores: list[float], unanswerable: list[bool]) -> tuple[float, float]:
    """Threshold t maximising refusal F1 when refusing every item scored below t; returns (t, F1)."""
    from vpqa.eval.metrics import refusal_prf

    best = (float("-inf"), 0.0)
    for t in sorted(set(scores)):
        f1 = refusal_prf([s < t for s in scores], unanswerable)[2]
        if f1 > best[1]:
            best = (t, f1)
    return best


def coverage_accuracy(answered: list[bool], correct: list[bool]) -> tuple[float, float]:
    """Share of items answered, and the share of answered items that are correct."""
    n = sum(answered)
    return n / len(answered), (sum(a and c for a, c in zip(answered, correct)) / n if n else 0.0)
