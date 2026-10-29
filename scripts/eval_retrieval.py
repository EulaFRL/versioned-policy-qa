"""Retrieval on MultiDoc2Dial: BM25, dense, RRF fusion, and cross-encoder rerank of the fused list,
under different query modes.

    python scripts/eval_retrieval.py --split validation --query-mode last gated oracle
    python scripts/eval_retrieval.py --query-mode gated --rerank 10 20 40
    python scripts/eval_retrieval.py --unit heading --query-mode history gated --rerank 40

Query modes: `last`, `history`, `recent:N` and `oracle:N` (N history utterances), `gated:W` (history
only for utterances of at most W words). See vpqa.retrieval.query.build_query.

`--rerank N` reorders the top N of the RRF list with the cross-encoder and leaves the rest in place.
Every candidate is scored once and cached under runs/rerank_cache/, so several N cost one pass.
`--rerank-query` gives the cross-encoder a different query mode from the retrievers, `--rerank-model` a
different cross-encoder (scripts/train_reranker.py; its scores get their own cache).

`--route router` restricts BM25 and dense to the domain predicted by the router from
scripts/train_router.py when it is confident; `--route oracle` uses the gold domain (an upper bound).

Besides `<split>.json`, the top 10 of every run per turn go to `<split>.ranked.json` for paired tests.

`--unit` picks the retrieval unit (vpqa.data.multidoc2dial). Besides R@k, every run reports Hit@k (any
gold passage in the top k, the paper's metric) and Ctx@W, the share of gold grounding spans inside the
first W words of retrieved text, which compares units of different sizes at the same reader budget.
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from collections import defaultdict
from pathlib import Path

import yaml

from vpqa.data.multidoc2dial import load_passages, load_turns, span_index
from vpqa.eval.metrics import hit_at_k, mrr, recall_at_k, span_recall_within
from vpqa.retrieval.bm25 import BM25Index
from vpqa.retrieval.dense import DenseIndex
from vpqa.retrieval.fusion import rrf
from vpqa.retrieval.query import build_query
from vpqa.retrieval.rerank import Reranker
from vpqa.retrieval.route import routing_text, thresholds_for

RAW = Path("data/raw/multidoc2dial")
KS = (1, 5, 10, 20)
BUDGETS = (200, 500)


def score(runs: list[list[str]], turns, ctx) -> dict:
    out = {f"R@{k}": sum(recall_at_k(r, t.gold_pids, k) for r, t in zip(runs, turns)) / len(turns) for k in KS}
    out["MRR@20"] = sum(mrr(r[:20], t.gold_pids) for r, t in zip(runs, turns)) / len(turns)
    out |= {f"Hit@{k}": sum(hit_at_k(r, t.gold_pids, k) for r, t in zip(runs, turns)) / len(turns) for k in KS}
    out |= {f"Ctx@{b}": sum(span_recall_within(r, t.gold_spans, *ctx, b) for r, t in zip(runs, turns)) / len(turns)
            for b in BUDGETS}
    out["n"] = len(turns)
    return out


def breakdown(runs, turns, ctx) -> dict:
    groups = defaultdict(list)
    for r, t in zip(runs, turns):
        groups["all"].append((r, t))
        groups[f"domain={t.domain}"].append((r, t))
        groups["first_turn" if not t.history else "follow_up"].append((r, t))
        groups["question" if (t.user_act or "").startswith("query") else "reply"].append((r, t))
    return {g: score([r for r, _ in xs], [t for _, t in xs], ctx) for g, xs in sorted(groups.items())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="validation")
    ap.add_argument("--unit", default="section", choices=["section", "heading", "token"])
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--query-mode", nargs="+", default=["last", "recent:4", "history"])
    ap.add_argument("--methods", nargs="+", default=["bm25", "dense", "rrf"])
    ap.add_argument("--rerank", nargs="*", type=int, default=[], help="rerank the top N of RRF")
    ap.add_argument("--rerank-query", help="query mode for the cross-encoder (default: same as retrieval)")
    ap.add_argument("--rerank-model", help="cross-encoder to rerank with (default: the config's), e.g. a fine-tuned one")
    ap.add_argument("--route", choices=["none", "router", "oracle"], default="none")
    ap.add_argument("--router", default="runs/router/router.pkl")
    ap.add_argument("--out", default="runs/retrieval")
    args = ap.parse_args()
    full_cfg = yaml.safe_load(open(args.config))
    cfg, qcfg, rcfg = full_cfg["retrieval"], full_cfg["query"], full_cfg["rerank"]

    doc_path = RAW / "multidoc2dial_doc.json"
    passages = load_passages(doc_path, args.unit)
    turns = [t for t in load_turns(RAW / f"multidoc2dial_dial_{args.split}.json", doc_path, args.unit) if t.answerable]
    print(f"{len(passages)} {args.unit} passages, {len(turns)} answerable turns ({args.split})")
    ctx = ({f"{d}::{sp}": pids for (d, sp), pids in span_index(doc_path, args.unit).items()},
           {p.pid: len(p.text.split()) for p in passages})

    bm25 = BM25Index(passages)
    dense = None
    if {"dense", "rrf"} & set(args.methods):
        name = cfg["dense_model"]
        unit = "" if args.unit == "section" else f"__{args.unit}"
        dense = DenseIndex(passages, name, cache=Path("data/index") / f"{name.replace('/', '__')}{unit}.npy")

    rerank_model = args.rerank_model or rcfg["model"]
    reranker = Reranker(passages, rerank_model) if args.rerank else None
    cache_name = args.split + ("" if rerank_model == rcfg["model"] else "__" + rerank_model.strip("/").replace("/", "__"))

    def queries_for(spec: str) -> list[str]:
        mode, _, n = spec.partition(":")
        history_turns, short_words = qcfg["history_turns"], qcfg["short_words"]
        if n:
            short_words = int(n) if mode == "gated" else short_words
            history_turns = int(n) if mode != "gated" else history_turns
        return [build_query(t, mode, history_turns, qcfg["max_words"], short_words) for t in turns]

    results, latency, routing, top10 = {}, {}, {}, {}
    domains: list[str | None] = [None] * len(turns)
    if args.route == "oracle":
        domains = [t.domain for t in turns]
    elif args.route == "router":
        with open(args.router, "rb") as f:
            saved = pickle.load(f)
        texts = [routing_text(t, saved["mode"]) for t in turns]
        th = thresholds_for(turns, saved["threshold"]["first_turn"], saved["threshold"]["follow_up"])
        domains = saved["router"].route(dense.encode(texts), th)
        t0 = time.perf_counter()
        for x, t in zip(texts[:200], th):  # one turn at a time
            saved["router"].route(dense.encode([x]), t)
        routing = {"threshold": saved["threshold"], "mode": saved["mode"],
                   "ms": (time.perf_counter() - t0) / 200 * 1000, **route_stats(domains, turns)}
        print("routing:", {k: round(v, 3) if isinstance(v, float) else v for k, v in routing.items()})
    suffix = ("" if args.route == "none" else f"+route:{args.route}") + ("" if args.unit == "section" else f"@{args.unit}")

    for spec in args.query_mode:
        queries = queries_for(spec)
        lat = {"query_words_mean": sum(len(q.split()) for q in queries) / len(queries)}
        t0 = time.perf_counter()
        ranked = {"bm25": [bm25.search(q, cfg["bm25_k"], d) for q, d in zip(queries, domains)]}
        lat["bm25_ms"] = (time.perf_counter() - t0) / len(queries) * 1000
        if dense is not None:
            sample = queries[:200]
            t0 = time.perf_counter()
            for q, d in zip(sample, domains):  # one query at a time, as at serving time
                dense.search(q, cfg["dense_k"], d)
            lat["dense_ms"] = (time.perf_counter() - t0) / len(sample) * 1000
            ranked["dense"] = dense.search_many(queries, cfg["dense_k"], domains)
            ranked["rrf"] = [rrf([b, d], k=cfg["rrf_k"]) for b, d in zip(ranked["bm25"], ranked["dense"])]
        if reranker is not None:
            rspec = args.rerank_query or spec
            rqueries = queries if rspec == spec else queries_for(rspec)
            scores = rerank_scores(reranker, rqueries, ranked["rrf"],
                                   Path("runs/rerank_cache") / f"{cache_name}.json")
            for n in args.rerank:
                name = f"rrf+rerank@{n}" + (f"[{rspec}]" if rspec != spec else "")
                ranked[name] = [sorted(c[:n], key=lambda p: -s[p]) + c[n:] for c, s in zip(ranked["rrf"], scores)]
                results[f"{name}/{spec}{suffix}"] = breakdown(ranked[name], turns, ctx)
                top10[f"{name}/{spec}{suffix}"] = [r[:10] for r in ranked[name]]
            n = max(args.rerank)
            t0 = time.perf_counter()
            for q, c in zip(rqueries[:100], ranked["rrf"][:100]):  # one query at a time
                reranker.rerank(q, c[:n])
            lat[f"rerank@{n}_ms"] = (time.perf_counter() - t0) / 100 * 1000
        latency[spec] = lat
        for m in args.methods:
            results[f"{m}/{spec}{suffix}"] = breakdown(ranked[m], turns, ctx)
            top10[f"{m}/{spec}{suffix}"] = [r[:10] for r in ranked[m]]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.split}.json").write_text(
        json.dumps({"results": results, "latency": latency, "routing": routing}, indent=2))
    (out / f"{args.split}.ranked.json").write_text(json.dumps(
        {"ids": [f"{t.dialog_id}/{t.turn_id}" for t in turns], "gold": [t.gold_pids for t in turns], "ranked": top10}))

    print(f"\n| run | " + " | ".join(f"R@{k}" for k in KS) + " | MRR@20 | follow-up R@5 | question R@5 |")
    print("|---" * (len(KS) + 4) + "|")
    for name, res in results.items():
        a = res["all"]
        print(f"| {name} | " + " | ".join(f"{a[f'R@{k}']:.3f}" for k in KS)
              + f" | {a['MRR@20']:.3f} | {res['follow_up']['R@5']:.3f} | {res['question']['R@5']:.3f} |")
    extra = [f"Hit@{k}" for k in KS[:3]] + [f"Ctx@{b}" for b in BUDGETS]
    print("\n| run | " + " | ".join(extra) + " |\n" + "|---" * (len(extra) + 1) + "|")
    for name, res in results.items():
        print(f"| {name} | " + " | ".join(f"{res['all'][m]:.3f}" for m in extra) + " |")
    print("\n| query | latency |\n|---|---|")
    for spec, lat in latency.items():
        print(f"| {spec} | " + ", ".join(f"{k} {v:.1f}" for k, v in lat.items()) + " |")


def route_stats(domains: list[str | None], turns) -> dict:
    """Share of turns routed, and accuracy of the routed ones, overall and on first turns."""
    out = {}
    for name, keep in [("all", lambda t: True), ("first_turn", lambda t: not t.history)]:
        sub = [(d, t) for d, t in zip(domains, turns) if keep(t)]
        routed = [(d, t) for d, t in sub if d is not None]
        out[f"{name}_coverage"] = len(routed) / len(sub)
        out[f"{name}_accuracy"] = sum(d == t.domain for d, t in routed) / max(len(routed), 1)
    return out


def rerank_scores(reranker: Reranker, queries: list[str], candidates: list[list[str]], cache: Path):
    """Cross-encoder score for every (query, candidate) pair, cached per query text as {pid: score}.
    Only pairs missing from the cache are scored, so runs with different candidate lists share it."""
    cached = json.loads(cache.read_text()) if cache.exists() else {}
    scores = [cached.setdefault(q, {}) for q in queries]
    missing = [(i, p) for i, c in enumerate(candidates) for p in c if p not in scores[i]]
    if missing:
        for (i, p), s in zip(missing, reranker.score_many([(queries[i], p) for i, p in missing])):
            scores[i][p] = s
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(".tmp")  # write then rename, so an interrupted run leaves the old cache intact
        tmp.write_text(json.dumps(cached))
        tmp.replace(cache)
    return scores


if __name__ == "__main__":
    main()
