from __future__ import annotations


def rrf(rankings: list[list[str]], k: int = 60, top_n: int | None = None) -> list[str]:
    """Reciprocal rank fusion over several ranked id lists."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, pid in enumerate(ranking):
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank + 1)
    fused = sorted(scores, key=lambda p: (-scores[p], p))
    return fused[:top_n] if top_n else fused
