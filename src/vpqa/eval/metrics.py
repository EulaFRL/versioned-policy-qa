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
