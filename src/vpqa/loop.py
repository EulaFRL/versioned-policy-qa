"""Offline improvement loop: attribute failures, try patches aimed at the largest failure category, and keep a
patch only if a paired test on frozen turns says it helps (scripts/loop.py).

A failure is a turn the system got wrong, in one of four categories, decided in this order:

- intent: an answerable turn the router sent to another department's documents;
- retrieval: an answerable turn whose gold passage is not among the passages shown to the answer model;
- generation: an answerable turn whose gold passage was shown, but the final reply does not cite it
  (a wrong answer, a citation of another passage, or a refusal);
- knowledge gap: an unanswerable turn (near-miss: the documents cover a neighbouring case only) that got an
  answer instead of a refusal. The fix belongs partly to the documents' owners; the system's part is to say
  that it does not know.

A patch changes configuration only (a parameter, or a rule line added to the answer prompt), so it can be
applied and rolled back without touching code.
"""
from __future__ import annotations

import copy
import json
import random
from collections import defaultdict
from math import comb
from pathlib import Path

CATEGORIES = ("intent", "retrieval", "generation", "knowledge gap")


def merge(cfg: dict, patch: dict) -> dict:
    """`cfg` with `patch` applied: nested dicts merge, lists are extended, other values replaced."""
    out = copy.deepcopy(cfg)
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v)
        elif isinstance(v, list) and isinstance(out.get(k), list):
            out[k] = out[k] + [x for x in v if x not in out[k]]
        else:
            out[k] = copy.deepcopy(v)
    return out


def read_trace(path: Path) -> dict[str, dict[str, dict]]:
    """run id -> node -> outputs (the first record of each node)."""
    out: dict[str, dict[str, dict]] = defaultdict(dict)
    for line in path.read_text().splitlines():
        r = json.loads(line)
        out[r["run_id"]].setdefault(r["node"], r.get("outputs") or {})
    return out


def success(row: dict, answerable: bool) -> bool:
    """Answerable: answered, citing a gold passage. Unanswerable: not answered."""
    if answerable:
        return row["outcome"] == "answered" and bool(set(row["final_pids"]) & set(row["gold"]))
    return row["outcome"] != "answered"


def attribute(row: dict, trace: dict[str, dict], answerable: bool, top_n: int) -> str | None:
    """The failure category of one turn, None if it succeeded."""
    if success(row, answerable):
        return None
    if not answerable:
        return "knowledge gap"
    routed, gold_domain = trace["route"].get("domain"), trace["<given>"]["turn"]["domain"]
    if routed is not None and routed != gold_domain:
        return "intent"
    if not set(row["gold"]) & set(trace["rerank"]["ranked"][:top_n]):
        return "retrieval"
    return "generation"


def mcnemar(a: list[bool], b: list[bool]) -> tuple[int, int, float]:
    """Turns where only A succeeds, only B succeeds, and the exact two-sided p-value."""
    only_a = sum(x and not y for x, y in zip(a, b))
    only_b = sum(y and not x for x, y in zip(a, b))
    n = only_a + only_b
    if n == 0:
        return 0, 0, 1.0
    tail = sum(comb(n, k) for k in range(min(only_a, only_b) + 1)) / 2 ** n
    return only_a, only_b, min(1.0, 2 * tail)


def sign_flip(a: list[float], b: list[float], samples: int = 20000, seed: int = 0) -> tuple[int, int, float]:
    """Paired test on per-turn success rates (each the mean over repeated samples): turns where A is higher,
    turns where B is higher, and the two-sided p-value of a random sign-flip permutation test on b - a."""
    d = [y - x for x, y in zip(a, b) if y != x]
    only_a, only_b = sum(x < 0 for x in d), sum(x > 0 for x in d)
    if not d:
        return 0, 0, 1.0
    rng, observed = random.Random(seed), abs(sum(d))
    hits = sum(abs(sum(x if rng.random() < 0.5 else -x for x in d)) >= observed - 1e-9 for _ in range(samples))
    return only_a, only_b, (hits + 1) / (samples + 1)


def bootstrap_diff(a: list[float], b: list[float], samples: int = 10000, seed: int = 0) -> tuple[float, float]:
    """95% paired bootstrap interval of mean(b) - mean(a)."""
    rng, n = random.Random(seed), len(a)
    diffs = sorted(sum(b[i] - a[i] for i in (rng.randrange(n) for _ in range(n))) / n for _ in range(samples))
    return diffs[int(0.025 * samples)], diffs[int(0.975 * samples) - 1]
