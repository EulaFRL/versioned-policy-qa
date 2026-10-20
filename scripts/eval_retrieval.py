"""Retrieval on MultiDoc2Dial: BM25, dense, RRF fusion, and cross-encoder rerank of the fused list,
under different query modes.

    python scripts/eval_retrieval.py --split validation --query-mode last gated oracle
    python scripts/eval_retrieval.py --query-mode gated --rerank 10 20 40

Query modes: `last`, `history`, `recent:N` and `oracle:N` (N history utterances), `gated:W` (history
only for utterances of at most W words). See vpqa.retrieval.query.build_query.

`--rerank N` reorders the top N of the RRF list with the cross-encoder and leaves the rest in place.
Every candidate is scored once and cached under runs/rerank_cache/, so several N cost one pass.
`--rerank-query` gives the cross-encoder a different query mode from the retrievers.
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
from vpqa.retrieval.rerank import Reranker

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
    ap.add_argument("--rerank", nargs="*", type=int, default=[], help="rerank the top N of RRF")
    ap.add_argument("--rerank-query", help="query mode for the cross-encoder (default: same as retrieval)")
    ap.add_argument("--out", default="runs/retrieval")
    args = ap.parse_args()
    full_cfg = yaml.safe_load(open(args.config))
    cfg, qcfg, rcfg = full_cfg["retrieval"], full_cfg["query"], full_cfg["rerank"]

    doc_path = RAW / "multidoc2dial_doc.json"
    passages = load_passages(doc_path)
    turns = [t for t in load_turns(RAW / f"multidoc2dial_dial_{args.split}.json", doc_path) if t.answerable]
    print(f"{len(passages)} passages, {len(turns)} answerable turns ({args.split})")

    bm25 = BM25Index(passages)
    dense = None
    if {"dense", "rrf"} & set(args.methods):
        name = cfg["dense_model"]
        dense = DenseIndex(passages, name, cache=Path("data/index") / f"{name.replace('/', '__')}.npy")

    reranker = Reranker(passages, rcfg["model"]) if args.rerank else None

    def queries_for(spec: str) -> list[str]:
        mode, _, n = spec.partition(":")
        history_turns, short_words = qcfg["history_turns"], qcfg["short_words"]
        if n:
            short_words = int(n) if mode == "gated" else short_words
            history_turns = int(n) if mode != "gated" else history_turns
        return [build_query(t, mode, history_turns, qcfg["max_words"], short_words) for t in turns]

    results, latency = {}, {}
    for spec in args.query_mode:
        queries = queries_for(spec)
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
        if reranker is not None:
            rspec = args.rerank_query or spec
            rqueries = queries if rspec == spec else queries_for(rspec)
            scores = rerank_scores(reranker, rqueries, ranked["rrf"],
                                   Path("runs/rerank_cache") / args.split / f"{spec}__{rspec}.json")
            for n in args.rerank:
                name = f"rrf+rerank@{n}" + (f"[{rspec}]" if rspec != spec else "")
                ranked[name] = [sorted(c[:n], key=lambda p: -s[p]) + c[n:] for c, s in zip(ranked["rrf"], scores)]
                results[f"{name}/{spec}"] = breakdown(ranked[name], turns)
            n = max(args.rerank)
            t0 = time.perf_counter()
            for q, c in zip(rqueries[:100], ranked["rrf"][:100]):  # one query at a time
                reranker.rerank(q, c[:n])
            lat[f"rerank@{n}_ms"] = (time.perf_counter() - t0) / 100 * 1000
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
    print("\n| query | latency |\n|---|---|")
    for spec, lat in latency.items():
        print(f"| {spec} | " + ", ".join(f"{k} {v:.1f}" for k, v in lat.items()) + " |")


def rerank_scores(reranker: Reranker, queries: list[str], candidates: list[list[str]], cache: Path):
    """Cross-encoder score for every (query, candidate) pair, cached as one {pid: score} per turn."""
    if cache.exists():
        scores = json.loads(cache.read_text())
        if len(scores) == len(queries) and all(set(c) <= set(s) for c, s in zip(candidates, scores)):
            return scores
    pairs = [(q, p) for q, c in zip(queries, candidates) for p in c]
    flat = iter(reranker.score_many(pairs))
    scores = [{p: next(flat) for p in c} for c in candidates]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(scores))
    return scores


if __name__ == "__main__":
    main()
