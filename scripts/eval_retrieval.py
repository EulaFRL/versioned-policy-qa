"""Retrieval baselines on MultiDoc2Dial: BM25, dense, and RRF fusion, under different query modes.

    python scripts/eval_retrieval.py --split validation --query-mode last recent:2 recent:4 history

Query modes: `last`, `history`, `recent:N` and `oracle:N` (N history utterances), `gated:W` (history
only for utterances of at most W words). See vpqa.retrieval.query.build_query.
"""
from __future__ import annotations

import argparse
import json
import time
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
        groups["question" if (t.user_act or "").startswith("query") else "reply"].append((r, t))
    return {g: score([r for r, _ in xs], [t for _, t in xs]) for g, xs in sorted(groups.items())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="validation")
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--query-mode", nargs="+", default=["last", "recent:4", "history"])
    ap.add_argument("--methods", nargs="+", default=["bm25", "dense", "rrf"])
    ap.add_argument("--out", default="runs/retrieval")
    args = ap.parse_args()
    full_cfg = yaml.safe_load(open(args.config))
    cfg, qcfg = full_cfg["retrieval"], full_cfg["query"]

    doc_path = RAW / "multidoc2dial_doc.json"
    passages = load_passages(doc_path)
    turns = [t for t in load_turns(RAW / f"multidoc2dial_dial_{args.split}.json", doc_path) if t.answerable]
    print(f"{len(passages)} passages, {len(turns)} answerable turns ({args.split})")

    bm25 = BM25Index(passages)
    dense = None
    if {"dense", "rrf"} & set(args.methods):
        name = cfg["dense_model"]
        dense = DenseIndex(passages, name, cache=Path("data/index") / f"{name.replace('/', '__')}.npy")

    results, latency = {}, {}
    for spec in args.query_mode:
        mode, _, n = spec.partition(":")
        history_turns, short_words = qcfg["history_turns"], qcfg["short_words"]
        if n:
            short_words = int(n) if mode == "gated" else short_words
            history_turns = int(n) if mode != "gated" else history_turns
        queries = [build_query(t, mode, history_turns, qcfg["max_words"], short_words) for t in turns]
        lat = {"query_words_mean": sum(len(q.split()) for q in queries) / len(queries)}
        t0 = time.perf_counter()
        ranked = {"bm25": [bm25.search(q, cfg["bm25_k"]) for q in queries]}
        lat["bm25_ms"] = (time.perf_counter() - t0) / len(queries) * 1000
        if dense is not None:
            sample = queries[:200]
            t0 = time.perf_counter()
            for q in sample:  # one query at a time, as at serving time
                dense.search(q, cfg["dense_k"])
            lat["dense_ms"] = (time.perf_counter() - t0) / len(sample) * 1000
            ranked["dense"] = dense.search_many(queries, cfg["dense_k"])
            ranked["rrf"] = [rrf([b, d], k=cfg["rrf_k"]) for b, d in zip(ranked["bm25"], ranked["dense"])]
        latency[spec] = lat
        for m in args.methods:
            results[f"{m}/{spec}"] = breakdown(ranked[m], turns)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.split}.json").write_text(json.dumps({"results": results, "latency": latency}, indent=2))

    print(f"\n| run | " + " | ".join(f"R@{k}" for k in KS) + " | MRR@20 | follow-up R@5 | question R@5 |")
    print("|---" * (len(KS) + 4) + "|")
    for name, res in results.items():
        a = res["all"]
        print(f"| {name} | " + " | ".join(f"{a[f'R@{k}']:.3f}" for k in KS)
              + f" | {a['MRR@20']:.3f} | {res['follow_up']['R@5']:.3f} | {res['question']['R@5']:.3f} |")
    print("\n| query | mean words | BM25 ms/query | dense ms/query |\n|---|---|---|---|")
    for spec, lat in latency.items():
        print(f"| {spec} | {lat['query_words_mean']:.1f} | {lat['bm25_ms']:.1f} | {lat.get('dense_ms', float('nan')):.1f} |")


if __name__ == "__main__":
    main()
