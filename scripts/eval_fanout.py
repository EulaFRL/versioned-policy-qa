"""Per-department retrieval on the turns the router does not route (vpqa.pipeline.dag, retrieve_departments).

    python scripts/eval_fanout.py --k 5 10

MultiDoc2Dial validation, answerable turns whose router confidence is below its threshold, so the
pipeline searches every domain. Compared on those turns, each list reranked in full:
- global: one search over all domains (bm25_k + dense_k candidates), the configured pipeline;
- departments, k: BM25 and dense top k inside each department, fused per department, then across;
- gold domain: one search inside the dialogue's own domain (upper bound for routing).
Reported: R@1 / R@5 / R@10, MRR@20, the number of candidates reranked, latency of retrieval and rerank,
and how many department searches missed their budget.
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import yaml

from vpqa.data.multidoc2dial import load_passages, load_turns
from vpqa.eval.metrics import mrr, recall_at_k
from vpqa.pipeline.dag import Pipeline

RAW = Path("data/raw/multidoc2dial")
DOC = RAW / "multidoc2dial_doc.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, nargs="+", default=[5, 10])
    ap.add_argument("--timeout-ms", type=float, default=250)
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--router", type=Path, default=Path("runs/router/router.pkl"))
    ap.add_argument("--out", type=Path, default=Path("runs/fanout"))
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    pipe = Pipeline(load_passages(DOC), cfg, Path("data/index/BAAI__bge-m3.npy"), args.router, timeouts=False)
    pipe.pool = ThreadPoolExecutor(len(pipe.departments))

    turns = [t for t in load_turns(RAW / "multidoc2dial_dial_validation.json", DOC) if t.answerable]
    queries = [pipe.query(t)["query"] for t in turns]
    domains = [pipe.route(t)["domain"] for t in turns]
    low = [(t, q) for t, q, d in zip(turns, queries, domains) if d is None]
    print(f"{len(low)} of {len(turns)} answerable validation turns are not routed")

    def search(q, t, setting):
        start = time.perf_counter()
        if setting == "global":
            out = pipe.retrieve(q, None, None)
        elif setting == "gold domain":
            out = pipe.retrieve(q, t.domain, None)
        else:
            pipe.fan_out = {"k": setting, "timeout_ms": args.timeout_ms}
            out = pipe.retrieve_departments(q, None)
        mid = time.perf_counter()
        ranked = pipe.rerank(q, out["candidates"])["ranked"]
        return ranked, out, (mid - start) * 1000, (time.perf_counter() - mid) * 1000

    settings = ["global", *args.k, "gold domain"]
    rows = {}
    for s in settings:
        res = [search(q, t, s) for t, q in low]
        golds = [t.gold_pids for t, _ in low]
        rows[f"departments, k={s}" if isinstance(s, int) else s] = {
            **{f"R@{k}": float(np.mean([recall_at_k(r, g, k) for (r, *_), g in zip(res, golds)])) for k in (1, 5, 10)},
            "MRR@20": float(np.mean([mrr(r[:20], g) for (r, *_), g in zip(res, golds)])),
            "candidates": float(np.mean([len(o["candidates"]) for _, o, _, _ in res])),
            "retrieve_ms_p50": float(np.median([a for *_, a, _ in res])),
            "rerank_ms_p50": float(np.median([b for *_, b in res])),
            "skipped_departments": int(sum(len(o["skipped"]) for _, o, _, _ in res)),
        }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "fanout.json").write_text(json.dumps({"turns": len(low), "results": rows}, indent=2))
    cols = ["R@1", "R@5", "R@10", "MRR@20", "candidates", "retrieve_ms_p50", "rerank_ms_p50", "skipped_departments"]
    lines = [f"{len(low)} unrouted turns of {len(turns)}", "", "| Search | " + " | ".join(cols) + " |",
             "|---" * (len(cols) + 1) + "|"]
    for name, m in rows.items():
        lines.append(f"| {name} | " + " | ".join(f"{m[c]:.3f}" if c.startswith(("R@", "MRR")) else f"{m[c]:.0f}"
                                                  for c in cols) + " |")
    (args.out / "fanout.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
