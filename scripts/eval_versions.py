"""Version-aware retrieval on the eCFR version-conflict questions (scripts/make_version_qa.py).

    python scripts/eval_versions.py --split dev
    python scripts/eval_versions.py --split dev --policy unaware latest prefilter postfilter --rerank

Every question carries an as-of date; its gold passage is the version of one paragraph in force on that
date, and the other versions of that paragraph are its stale (or not yet in force) competitors.
Version policies:
- `unaware`: one index over all versions, the date is ignored;
- `latest`: index only the versions in force today (what a system that keeps the current text does);
- `prefilter[:N]`: BM25 and dense each return their top N (default 20) among the versions in force on
  the as-of date;
- `postfilter:N`: each retriever returns its top N over all versions, then versions not in force are dropped.

The top 10 of every run, per question, go to `<name>.ranked.json` for paired tests (scripts/sig_tests.py).

Metrics: R@k (gold version in the top k), Key@k (any version of the gold paragraph in the top k),
Stale@1 (the top passage is another version of the gold paragraph) and Invalid@5 (share of the top 5
passages not in force on the as-of date).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np

from vpqa.data.ecfr import load_versions
from vpqa.eval.metrics import recall_at_k
from vpqa.retrieval.bm25 import BM25Index, tokenize
from vpqa.retrieval.dense import DenseIndex
from vpqa.retrieval.fusion import rrf
from vpqa.retrieval.rerank import Reranker

KS = (1, 5, 10)
DEPTH = 20


def in_force(passages, on: date) -> np.ndarray:
    return np.array([p.effective_from <= on and (p.effective_to is None or on < p.effective_to) for p in passages])


def top(scores: np.ndarray, pids: list[str], k: int, mask: np.ndarray | None = None) -> list[str]:
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)
    order = np.argsort(-scores, kind="stable")[:k]
    return [pids[i] for i in order if np.isfinite(scores[i])]


def score(runs: list[list[str]], items: list[dict], valid: dict[str, set[str]]) -> dict:
    key = lambda pid: pid.split("@")[0]
    out = {f"R@{k}": float(np.mean([recall_at_k(r, [it["gold_pid"]], k) for r, it in zip(runs, items)])) for k in KS}
    out["Key@5"] = float(np.mean([any(key(p) == key(it["gold_pid"]) for p in r[:5]) for r, it in zip(runs, items)]))
    out["Stale@1"] = float(np.mean([bool(r) and r[0] in it["other_versions"] for r, it in zip(runs, items)]))
    out["Invalid@5"] = float(np.mean([np.mean([p not in valid[it["qid"]] for p in r[:5]]) if r else 0.0
                                      for r, it in zip(runs, items)]))
    out["n"] = len(items)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", type=Path, default=Path("data/benchmark/ecfr_version_qa.jsonl"))
    ap.add_argument("--versions", type=Path, default=Path("data/processed/ecfr_versions.jsonl"))
    ap.add_argument("--split", default="dev")
    ap.add_argument("--policy", nargs="+", default=["unaware", "latest", "prefilter", "postfilter:20", "postfilter:100"])
    ap.add_argument("--rerank", action="store_true", help="also rerank the fused list with the cross-encoder")
    ap.add_argument("--out", default="runs/versions")
    args = ap.parse_args()

    passages = load_versions(args.versions)
    pids = [p.pid for p in passages]
    items = [json.loads(l) for l in args.qa.read_text().splitlines()]
    items = [it for it in items if it["split"] == args.split]
    print(f"{len(passages)} passage versions, {len(items)} {args.split} items")

    bm25 = BM25Index(passages)
    dense = DenseIndex(passages, cache=Path("data/index/BAAI__bge-m3__ecfr.npy"))
    reranker = Reranker(passages) if args.rerank else None
    q_emb = dense.encode([it["question"] for it in items])
    dense_scores = q_emb @ dense.emb.T
    bm25_scores = [bm25.bm25.get_scores(tokenize(it["question"])) for it in items]
    current = np.array([p.effective_to is None for p in passages])
    valid_masks = [in_force(passages, date.fromisoformat(it["as_of"])) for it in items]
    valid = {it["qid"]: {pids[i] for i in np.flatnonzero(m)} for it, m in zip(items, valid_masks)}

    results, ranked = {}, {}
    for policy in args.policy:
        name, _, n = policy.partition(":")
        runs = defaultdict(list)
        for i, it in enumerate(items):
            if name == "unaware":
                mask, depth = None, DEPTH
            elif name == "latest":
                mask, depth = current, DEPTH
            elif name == "prefilter":
                mask, depth = valid_masks[i], int(n or DEPTH)
            elif name == "postfilter":
                mask, depth = None, int(n)
            else:
                raise ValueError(policy)
            lists = {"bm25": top(bm25_scores[i], pids, depth, mask), "dense": top(dense_scores[i], pids, depth, mask)}
            if name == "postfilter":
                lists = {m: [p for p in l if p in valid[it["qid"]]] for m, l in lists.items()}
            lists["rrf"] = rrf([lists["bm25"], lists["dense"]])
            if reranker is not None:
                lists["rrf+rerank"] = reranker.rerank(it["question"], lists["rrf"])
            for m, l in lists.items():
                runs[m].append(l)
        for m, r in runs.items():
            ranked[f"{policy} / {m}"] = [x[:10] for x in r]
            results[f"{policy} / {m}"] = {"all": score(r, items, valid),
                                          **{f"version={v}": score([x for x, it in zip(r, items) if it["version"] == v],
                                                                   [it for it in items if it["version"] == v], valid)
                                             for v in ("old", "new")}}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    name = "_".join([args.split, *(p.replace(":", "") for p in args.policy)] + (["rerank"] if args.rerank else []))
    (out / f"{name}.json").write_text(json.dumps(results, indent=2))
    (out / f"{name}.ranked.json").write_text(json.dumps({"qids": [it["qid"] for it in items], "ranked": ranked}))
    cols = [f"R@{k}" for k in KS] + ["Key@5", "Stale@1", "Invalid@5"]
    print("| policy / retriever | " + " | ".join(cols) + " | old R@1 | new R@1 |")
    print("|---" * (len(cols) + 3) + "|")
    for name, r in results.items():
        print(f"| {name} | " + " | ".join(f"{r['all'][c]:.3f}" for c in cols)
              + f" | {r['version=old']['R@1']:.3f} | {r['version=new']['R@1']:.3f} |")


if __name__ == "__main__":
    main()
