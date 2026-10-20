from __future__ import annotations


def recall_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    if not gold:
        return 0.0
    hit = set(retrieved[:k]) & set(gold)
    return len(hit) / len(set(gold))


def mrr(retrieved: list[str], gold: list[str]) -> float:
    gold_set = set(gold)
    for rank, pid in enumerate(retrieved, start=1):
        if pid in gold_set:
            return 1.0 / rank
    return 0.0


def refusal_prf(pred_refused: list[bool], unanswerable: list[bool]) -> tuple[float, float, float]:
    tp = sum(p and u for p, u in zip(pred_refused, unanswerable))
    fp = sum(p and not u for p, u in zip(pred_refused, unanswerable))
    fn = sum(u and not p for p, u in zip(pred_refused, unanswerable))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def hit_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    """1 if any gold passage is in the top k (the MultiDoc2Dial paper's R@k)."""
    return float(bool(set(retrieved[:k]) & set(gold)))


def span_recall_within(retrieved: list[str], gold_spans: list[str], span_pids: dict[str, list[str]],
                       words: dict[str, int], budget: int) -> float:
    """Share of gold grounding spans inside the context a reader would get: whole passages in rank order
    until the next one would exceed `budget` words. Comparable across retrieval units of any size."""
    if not gold_spans:
        return 0.0
    taken, used = set(), 0
    for pid in retrieved:
        if used + words[pid] > budget:
            break
        taken.add(pid)
        used += words[pid]
    return sum(any(p in taken for p in span_pids[s]) for s in gold_spans) / len(gold_spans)
