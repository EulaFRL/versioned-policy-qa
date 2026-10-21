"""Cited answers through the full DAG on a fixed-seed sample of turns.

    python scripts/eval_answers.py --dataset md2d --n 20
    python scripts/eval_answers.py --dataset ecfr --n 20

Needs VPQA_API_BASE / VPQA_API_KEY / VPQA_MODEL (vpqa.llm). Per item it records the answer, the cited pids
and the parse issues; the summary reports how often the model refused, cited a gold passage (Cite-Hit),
the share of cited passages that are gold (Cite-Prec), how often the reply had a parse issue, how often
the verifier passed it, latency
and the API cost of the run (cached calls count at the cost of the call that filled the cache).
"""
from __future__ import annotations

import argparse
import json
import random
from datetime import date
from pathlib import Path

import numpy as np
import yaml

from vpqa import llm
from vpqa.data.ecfr import load_versions
from vpqa.data.multidoc2dial import load_passages, load_turns
from vpqa.pipeline.dag import Pipeline
from vpqa.schema import Turn

RAW = Path("data/raw/multidoc2dial")


def load(dataset: str, cfg: dict, args) -> tuple[Pipeline, list[Turn]]:
    if dataset == "md2d":
        doc = RAW / "multidoc2dial_doc.json"
        turns = [t for t in load_turns(RAW / f"multidoc2dial_dial_{args.split}.json", doc) if t.answerable]
        pipe = Pipeline(load_passages(doc), cfg, Path("data/index/BAAI__bge-m3.npy"), args.router, generate=True)
    else:
        items = [json.loads(l) for l in args.qa.read_text().splitlines()]
        split = "dev" if args.split == "validation" else args.split
        turns = [Turn(it["qid"], 0, "", [], it["question"], [it["gold_pid"]], as_of=date.fromisoformat(it["as_of"]))
                 for it in items if it["split"] == split]
        pipe = Pipeline(load_versions(), cfg, Path("data/index/BAAI__bge-m3__ecfr.npy"), generate=True)
    return pipe, random.Random(args.seed).sample(turns, min(args.n, len(turns)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["md2d", "ecfr"], default="md2d")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--router", type=Path, default=Path("runs/router/router.pkl"))
    ap.add_argument("--qa", type=Path, default=Path("data/benchmark/ecfr_version_qa.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("runs/answers"))
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    pipe, turns = load(args.dataset, cfg, args)

    args.out.mkdir(parents=True, exist_ok=True)
    name = f"{args.dataset}_{args.split}_n{args.n}_s{args.seed}"
    trace_path = args.out / f"{name}.trace.jsonl"
    trace_path.unlink(missing_ok=True)
    rows = []
    for t in turns:
        state, trace = pipe.run(t)
        run_id = f"{t.dialog_id}/{t.turn_id}"
        trace.write(trace_path, run_id)
        ms = {r["node"]: r["ms"] for r in trace.records}
        status = {r["node"]: r["status"] for r in trace.records if r["status"] != "ok"}
        rows.append({"id": run_id, "query": t.query, "gold": t.gold_pids, "top5": state["ranked"][:5],
                     **{k: state[k] for k in ("answer", "cited_pids", "refused", "issues", "verified", "problems")},
                     "answer_ms": ms.get("answer"), "degraded": status})
    (args.out / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    answered = [r for r in rows if not r["refused"]]
    gold_in_top = [r for r in rows if set(r["gold"]) & set(r["top5"])]
    summary = {
        "n": len(rows),
        "refused": sum(r["refused"] for r in rows) / len(rows),
        "refused_when_gold_in_top5": sum(r["refused"] for r in gold_in_top) / max(len(gold_in_top), 1),
        "Cite-Hit": sum(bool(set(r["cited_pids"]) & set(r["gold"])) for r in answered) / max(len(answered), 1),
        "Cite-Prec": float(np.mean([np.mean([p in r["gold"] for p in r["cited_pids"]])
                                    for r in answered if r["cited_pids"]] or [0.0])),
        "with_issue": sum(bool(r["issues"]) for r in answered) / max(len(answered), 1),
        "verified": sum(r["verified"] for r in answered) / max(len(answered), 1),
        "degraded": sum(bool(r["degraded"]) for r in rows) / len(rows),
        "answer_ms_p50": float(np.median([r["answer_ms"] for r in rows if r["answer_ms"] is not None])),
        "llm": dict(llm.TOTALS),
        "cost_per_item": llm.TOTALS["cost"] / len(rows),
    }
    (args.out / f"{name}.summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
