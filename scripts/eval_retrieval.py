"""Retrieval baselines on MultiDoc2Dial: BM25, dense, and RRF fusion, under different query modes.

    python scripts/eval_retrieval.py --split validation --query-mode last history
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import yaml

from vpqa.data.multidoc2dial import load_passages, load_turns
from vpqa.eval.metrics import mrr, recall_at_k
from vpqa.retrieval.bm25 import BM25Index
from vpqa.retrieval.dense import DenseIndex
from vpqa.retrieval.fusion import rrf
from vpqa.retrieval.query import build_query

RAW = Path("data/raw/multidoc2dial")
KS = (1, 5, 10, 20)


def score(runs: list[list[str]], turns) -> dict:
    out = {f"R@{k}": sum(recall_at_k(r, t.gold_pids, k) for r, t in zip(runs, turns)) / len(turns) for k in KS}
    out["MRR@20"] = sum(mrr(r[:20], t.gold_pids) for r, t in zip(runs, turns)) / len(turns)
    out["n"] = len(turns)
    return out


def breakdown(runs, turns) -> dict:
    groups = defaultdict(list)
    for r, t in zip(runs, turns):
        groups["all"].append((r, t))
        groups[f"domain={t.domain}"].append((r, t))
        groups["first_turn" if not t.history else "follow_up"].append((r, t))
    return {g: score([r for r, _ in xs], [t for _, t in xs]) for g, xs in sorted(groups.items())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="validation")
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--query-mode", nargs="+", default=["last", "history"])
    ap.add_argument("--methods", nargs="+", default=["bm25", "dense", "rrf"])
    ap.add_argument("--out", default="runs/retrieval")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))["retrieval"]

    doc_path = RAW / "multidoc2dial_doc.json"
    passages = load_passages(doc_path)
    turns = [t for t in load_turns(RAW / f"multidoc2dial_dial_{args.split}.json", doc_path) if t.answerable]
    print(f"{len(passages)} passages, {len(turns)} answerable turns ({args.split})")

    bm25 = BM25Index(passages)
    dense = None
    if {"dense", "rrf"} & set(args.methods):
        name = cfg["dense_model"]
        dense = DenseIndex(passages, name, cache=Path("data/index") / f"{name.replace('/', '__')}.npy")

    results = {}
    for mode in args.query_mode:
        queries = [build_query(t, mode) for t in turns]
        ranked = {"bm25": [bm25.search(q, cfg["bm25_k"]) for q in queries]}
        if dense is not None:
            ranked["dense"] = dense.search_many(queries, cfg["dense_k"])
            ranked["rrf"] = [rrf([b, d], k=cfg["rrf_k"]) for b, d in zip(ranked["bm25"], ranked["dense"])]
        for m in args.methods:
            results[f"{m}/{mode}"] = breakdown(ranked[m], turns)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.split}.json").write_text(json.dumps(results, indent=2))

    print(f"\n| run | " + " | ".join(f"R@{k}" for k in KS) + " | MRR@20 | follow-up R@5 |")
    print("|---" * (len(KS) + 3) + "|")
    for name, res in results.items():
        a = res["all"]
        print(f"| {name} | " + " | ".join(f"{a[f'R@{k}']:.3f}" for k in KS)
              + f" | {a['MRR@20']:.3f} | {res['follow_up']['R@5']:.3f} |")


if __name__ == "__main__":
    main()
