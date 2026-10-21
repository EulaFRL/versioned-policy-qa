"""Check that the retrieval DAG (vpqa.pipeline.dag) reproduces the evaluation scripts.

    python scripts/check_dag.py --md2d 50

MultiDoc2Dial: the first N validation turns go through the DAG and through the batch code path of
scripts/eval_retrieval.py (gated query, router, RRF, rerank of all candidates); the ranked lists must match.
eCFR: every dev question goes through the DAG with its as-of date; R@1 and R@5 must match the `prefilter`
row of scripts/eval_versions.py --rerank. Both checks run without stage timeouts, so nothing degrades.
Degraded: the first 20 MultiDoc2Dial turns with every stage budget set to 1 ms must come back with all
domains searched, BM25 candidates only, and no rerank.
"""
from __future__ import annotations

import argparse
import json
import pickle
from datetime import date
from pathlib import Path

import yaml

from vpqa.data.ecfr import load_versions
from vpqa.data.multidoc2dial import load_passages, load_turns
from vpqa.eval.metrics import recall_at_k
from vpqa.pipeline.dag import Pipeline
from vpqa.retrieval.fusion import rrf
from vpqa.retrieval.query import build_query
from vpqa.retrieval.route import routing_text, thresholds_for
from vpqa.schema import Turn

RAW = Path("data/raw/multidoc2dial")


def check_md2d(cfg: dict, n: int, router: Path, trace: Path) -> None:
    doc = RAW / "multidoc2dial_doc.json"
    passages = load_passages(doc, "section")
    turns = [t for t in load_turns(RAW / "multidoc2dial_dial_validation.json", doc, "section") if t.answerable][:n]
    pipe = Pipeline(passages, cfg, Path("data/index/BAAI__bge-m3.npy"), router, timeouts=False)
    q, r = cfg["query"], cfg["retrieval"]

    # reference: the batch path of eval_retrieval.py
    queries = [build_query(t, "gated", q["history_turns"], q["max_words"], q["short_words"]) for t in turns]
    with open(router, "rb") as f:
        saved = pickle.load(f)
    th = thresholds_for(turns, saved["threshold"]["first_turn"], saved["threshold"]["follow_up"])
    domains = saved["router"].route(pipe.dense.encode([routing_text(t, saved["mode"]) for t in turns]), th)
    bm25 = [pipe.bm25.search(x, r["bm25_k"], d) for x, d in zip(queries, domains)]
    dense = pipe.dense.search_many(queries, r["dense_k"], domains)
    ref = [pipe.reranker.rerank(x, rrf([b, d], k=r["rrf_k"])) for x, b, d in zip(queries, bm25, dense)]

    same = top1 = 0
    for i, (t, want) in enumerate(zip(turns, ref)):
        state, tr = pipe.run(t)
        tr.write(trace, f"md2d/{t.dialog_id}/{t.turn_id}")
        same += state["ranked"] == want
        top1 += state["ranked"][:5] == want[:5]
    print(f"MultiDoc2Dial: {same}/{len(turns)} identical ranked lists, {top1}/{len(turns)} identical top 5")

    tight = {**cfg, "harness": {"timeouts_ms": {"route": 1, "retrieve": 1, "rerank": 1}}}
    pipe = Pipeline(passages, tight, Path("data/index/BAAI__bge-m3.npy"), router)
    ok = 0
    for t, x in zip(turns[:20], queries):
        state, tr = pipe.run(t)
        tr.write(trace, f"degraded/{t.dialog_id}/{t.turn_id}")
        status = {r["node"]: r["status"] for r in tr.records}
        ok += (state["domain"] is None and state["ranked"] == pipe.bm25.search(x, r["bm25_k"])
               and status["route"] == status["retrieve"] == status["rerank"] == "timeout")
    print(f"Degraded: {ok}/20 turns fell back at every budgeted stage")


def check_ecfr(cfg: dict, qa: Path, trace: Path) -> None:
    passages = load_versions()
    items = [json.loads(l) for l in qa.read_text().splitlines()]
    items = [it for it in items if it["split"] == "dev"]
    pipe = Pipeline(passages, cfg, Path("data/index/BAAI__bge-m3__ecfr.npy"), timeouts=False)
    runs = []
    for it in items:
        turn = Turn(it["qid"], 0, "", [], it["question"], [it["gold_pid"]], as_of=date.fromisoformat(it["as_of"]))
        state, tr = pipe.run(turn)
        tr.write(trace, f"ecfr/{it['qid']}")
        runs.append(state["ranked"])
    r = {k: sum(recall_at_k(x, [it["gold_pid"]], k) for x, it in zip(runs, items)) / len(items) for k in (1, 5)}
    print(f"eCFR dev ({len(items)} items): R@1 {r[1]:.3f}, R@5 {r[5]:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--md2d", type=int, default=50)
    ap.add_argument("--router", type=Path, default=Path("runs/router/router.pkl"))
    ap.add_argument("--qa", type=Path, default=Path("data/benchmark/ecfr_version_qa.jsonl"))
    ap.add_argument("--trace", type=Path, default=Path("runs/traces/check_dag.jsonl"))
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    args.trace.parent.mkdir(parents=True, exist_ok=True)
    args.trace.unlink(missing_ok=True)
    check_md2d(cfg, args.md2d, args.router, args.trace)
    check_ecfr(cfg, args.qa, args.trace)


if __name__ == "__main__":
    main()
