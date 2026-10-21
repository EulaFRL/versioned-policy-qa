"""Stage budgets under pressure: the retrieval DAG with every budget scaled down, on Apple MPS and on CPU.

    python scripts/eval_budgets.py --n 300 --scale 1 0.5 0.25 0.1
    python scripts/eval_budgets.py --n 300 --scale 1 0.5 --device cpu

The 300 MultiDoc2Dial validation turns of scripts/eval_answers.py (same seed). Each setting multiplies the
configured route / retrieve / rerank budgets (configs `harness`); "none" runs without budgets. A stage
over budget falls back (route -> all domains, retrieve -> BM25 only, rerank -> RRF order). Reported per
setting: the share of turns where each stage fell back, R@1 / R@5 / MRR@20 of the final ranking, and
end-to-end retrieval latency (p50, p95). Timed-out stages keep running in their thread and hold the model
lock, so the next turn can find the lock busy: that cost is part of what is measured.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import yaml

from vpqa.data.multidoc2dial import load_passages, load_turns
from vpqa.eval.metrics import mrr, recall_at_k
from vpqa.pipeline.dag import Pipeline

RAW = Path("data/raw/multidoc2dial")
DOC = RAW / "multidoc2dial_doc.json"
STAGES = ("route", "retrieve", "rerank")


def to_cpu(pipe: Pipeline) -> None:
    import torch
    pipe.dense.model.to("cpu")
    pipe.reranker.model.to("cpu")
    pipe.reranker.model.to(torch.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale", nargs="+", default=["none", "1", "0.5", "0.25", "0.1"])
    ap.add_argument("--device", choices=["mps", "cpu"], default="mps")
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--router", type=Path, default=Path("runs/router/router.pkl"))
    ap.add_argument("--out", type=Path, default=Path("runs/budgets"))
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    base = {k: v for k, v in cfg["harness"]["timeouts_ms"].items() if k in STAGES}

    turns = [t for t in load_turns(RAW / "multidoc2dial_dial_validation.json", DOC) if t.answerable]
    turns = random.Random(args.seed).sample(turns, min(args.n, len(turns)))
    passages = load_passages(DOC)
    results = {}
    for scale in args.scale:
        budgets = {} if scale == "none" else {k: v * float(scale) for k, v in base.items()}
        pipe = Pipeline(passages, {**cfg, "harness": {"timeouts_ms": budgets}}, Path("data/index/BAAI__bge-m3.npy"),
                        args.router)
        if args.device == "cpu":
            to_cpu(pipe)
            pipe.warm_up()
        ranked, fell_back, ms = [], {s: 0 for s in STAGES}, []
        for t in turns:
            start = time.perf_counter()
            state, trace = pipe.run(t)
            ms.append((time.perf_counter() - start) * 1000)
            ranked.append(state["ranked"])
            for r in trace.records:
                if r["node"] in STAGES and r["status"] != "ok":
                    fell_back[r["node"]] += 1
        with pipe.model_lock:  # let abandoned stages finish before the next setting
            pass
        golds = [t.gold_pids for t in turns]
        results[scale] = {
            "budgets_ms": budgets,
            **{f"fallback_{s}": c / len(turns) for s, c in fell_back.items()},
            **{f"R@{k}": float(np.mean([recall_at_k(r, g, k) for r, g in zip(ranked, golds)])) for k in (1, 5)},
            "MRR@20": float(np.mean([mrr(r[:20], g) for r, g in zip(ranked, golds)])),
            "ms_p50": float(np.median(ms)), "ms_p95": float(np.percentile(ms, 95)),
        }
        print(scale, json.dumps(results[scale]))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"budgets_{args.device}_n{args.n}.json").write_text(json.dumps(results, indent=2))
    cols = ["fallback_route", "fallback_retrieve", "fallback_rerank", "R@1", "R@5", "MRR@20", "ms_p50", "ms_p95"]
    lines = [f"{args.device}, {len(turns)} turns", "", "| Budget scale | " + " | ".join(cols) + " |",
             "|---" * (len(cols) + 1) + "|"]
    for scale, m in results.items():
        lines.append(f"| {scale} | " + " | ".join(f"{m[c]:.0f}" if c.startswith("ms") else f"{m[c]:.3f}" for c in cols)
                     + " |")
    (args.out / f"budgets_{args.device}_n{args.n}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
