"""Cited answers through the full DAG on a fixed-seed sample of turns.

    python scripts/eval_answers.py --dataset md2d --n 20
    python scripts/eval_answers.py --dataset ecfr --n 20
    python scripts/eval_answers.py --dataset md2d --n 300 --withhold gold-doc
    python scripts/eval_answers.py --dataset ecfr --split dev --n 100 --versions unaware
    python scripts/eval_answers.py --dataset ecfr --split dev --n 100 --versions inject
    python scripts/eval_answers.py --dataset md2d --n 300 --withhold near-miss
    python scripts/eval_answers.py --dataset md2d --n 300 --check-question

--withhold gold-doc turns the same sample into unanswerable questions: every passage of the documents that
hold a gold passage is masked out of retrieval for that turn, so the right reply is a refusal. The two runs
pair 1:1 by id (scripts/refusal_report.py). --withhold near-miss keeps the gold documents but replaces the
latest user message with its near-miss rewrite (scripts/make_near_miss.py), on the turns whose rewrite was
kept, so retrieval finds the right page and the page does not answer.

--check-question turns on the verifier's check that the answer replies to what was asked
(configs `answer.check_question`); output names get `_cq`.

--versions (eCFR) sets how stale versions can reach the answer: `filter`, retrieval keeps only versions in
force on the as-of date (the pipeline as configured); `unaware`, retrieval searches every version; `inject`,
as `filter`, then another version of the gold paragraph, not in force on the as-of date, is put first among
the passages shown. The verifier always checks cited versions against the as-of date. The summary adds
the share of answers citing a version not in force, before and after the verifier, and how many of the
first answers that did were rejected.

Needs VPQA_API_BASE / VPQA_API_KEY / VPQA_MODEL (vpqa.llm). Per item it records the first answer, its
verdict, the number of revisions and the final outcome (answered, refused, or degraded to passage text).
The summary reports, for the first answer: how often the model refused, cited a gold passage (Cite-Hit),
the share of cited passages that are gold (Cite-Prec), how often the reply had a parse issue and how
often the verifier passed it; for the final output: the outcome shares, Cite-Hit, Cite-Hit on the items
whose gold passage was in the top 5 shown to the model (the generation side alone), and on MultiDoc2Dial
Span-Rec, the share of gold grounding spans inside a cited passage; latency
and the API cost of the run (cached calls count at the cost of the call that filled the cache).
"""
from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import yaml

from vpqa import llm
from vpqa.data.ecfr import load_versions
from vpqa.data.multidoc2dial import load_passages, load_turns, span_index
from vpqa.eval.refusal import withhold_gold_docs
from vpqa.pipeline import verifier
from vpqa.pipeline.dag import Pipeline
from vpqa.pipeline.engine import Graph, Node
from vpqa.schema import Turn

RAW = Path("data/raw/multidoc2dial")


def load(dataset: str, cfg: dict, args) -> tuple[Pipeline, list[Turn]]:
    if dataset == "md2d":
        doc = RAW / "multidoc2dial_doc.json"
        turns = [t for t in load_turns(RAW / f"multidoc2dial_dial_{args.split}.json", doc) if t.answerable]
        pipe = Pipeline(load_passages(doc), cfg, Path("data/index/BAAI__bge-m3.npy"), args.router, generate=True,
                        timeouts=args.workers == 1)
    else:
        items = [json.loads(l) for l in args.qa.read_text().splitlines()]
        split = "dev" if args.split == "validation" else args.split
        turns = [Turn(it["qid"], 0, "", [], it["question"], [it["gold_pid"]], as_of=date.fromisoformat(it["as_of"]))
                 for it in items if it["split"] == split]
        pipe = Pipeline(load_versions(), cfg, Path("data/index/BAAI__bge-m3__ecfr.npy"), generate=True,
                        version_filter=args.versions != "unaware", timeouts=args.workers == 1)
        if args.versions == "inject":
            stale = {it["qid"]: it["other_versions"] for it in items}
            inject_stale(pipe, stale)
    turns = random.Random(args.seed).sample(turns, min(args.n, len(turns)))
    if args.withhold == "gold-doc":
        turns = withhold_gold_docs(turns, pipe.by_pid)
    elif args.withhold == "near-miss":
        rewrites = {r["id"]: r for r in map(json.loads, (args.near_miss / f"md2d_near_miss_{args.split}.jsonl")
                                            .read_text().splitlines()) if r["kept"]}
        turns = [replace(t, query=rewrites[f"{t.dialog_id}/{t.turn_id}"]["question"], answerable=False)
                 for t in turns if f"{t.dialog_id}/{t.turn_id}" in rewrites]
    return pipe, turns


def inject_stale(pipe: Pipeline, stale: dict[str, list[str]]) -> None:
    """Insert a node after rerank that puts a version not in force on the as-of date first."""
    def inject(turn: Turn, ranked: list[str]) -> dict:
        old = [p for p in stale[turn.dialog_id] if not verifier.in_force(pipe.by_pid[p], turn.as_of)]
        return {"ranked": old[:1] + [p for p in ranked if p not in old[:1]]}

    nodes = pipe.graph.nodes
    at = [n.name for n in nodes].index("rerank") + 1
    node = Node("inject_stale", inject, {"turn": Turn, "ranked": list[str]}, {"ranked": list[str]})
    pipe.graph = Graph(nodes[:at] + [node] + nodes[at:], given=pipe.graph.given)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["md2d", "ecfr"], default="md2d")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--router", type=Path, default=Path("runs/router/router.pkl"))
    ap.add_argument("--qa", type=Path, default=Path("data/benchmark/ecfr_version_qa.jsonl"))
    ap.add_argument("--withhold", choices=["none", "gold-doc", "near-miss"], default="none")
    ap.add_argument("--near-miss", type=Path, default=Path("data/benchmark"))
    ap.add_argument("--check-question", action="store_true")
    ap.add_argument("--versions", choices=["filter", "unaware", "inject"], default="filter")
    ap.add_argument("--out", type=Path, default=Path("runs/answers"))
    ap.add_argument("--workers", type=int, default=1,
                    help="turns run at once; above 1, no stage budgets and no latency (threads wait on each other)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.check_question:
        cfg.setdefault("answer", {})["check_question"] = True
    pipe, turns = load(args.dataset, cfg, args)
    span_pids = ({f"{d}::{sp}": pids for (d, sp), pids in span_index(RAW / "multidoc2dial_doc.json").items()}
                 if args.dataset == "md2d" else {})

    args.out.mkdir(parents=True, exist_ok=True)
    name = (f"{args.dataset}_{args.split}_n{args.n}_s{args.seed}"
            + {"none": "", "gold-doc": "_withheld", "near-miss": "_nearmiss"}[args.withhold]
            + (f"_{args.versions}" if args.versions != "filter" else "") + ("_cq" if args.check_question else ""))
    trace_path = args.out / f"{name}.trace.jsonl"
    trace_path.unlink(missing_ok=True)
    rows = []
    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(pipe.run, turns))
    for t, (state, trace) in zip(turns, results):
        run_id = f"{t.dialog_id}/{t.turn_id}"
        trace.write(trace_path, run_id)
        ms = {r["node"]: r["ms"] for r in trace.records}
        status = {r["node"]: r["status"] for r in trace.records if r["status"] not in ("ok", "skipped")}
        first = {r["node"]: r.get("outputs", {}) for r in trace.records}
        stale = lambda pids: [p for p in pids if not verifier.in_force(pipe.by_pid[p], t.as_of)]
        rows.append({"id": run_id, "query": t.query, "gold": t.gold_pids, "gold_spans": t.gold_spans,
                     "top5": state["ranked"][:5],
                     "answer": first["answer"]["answer"], "cited_pids": first["answer"]["cited_pids"],
                     "refused": first["answer"]["refused"], "issues": first["answer"]["issues"],
                     "verified": first["verify"]["verified"], "problems": first["verify"]["problems"],
                     "revisions": sum(n.startswith("revise") and ms[n] > 0 for n in ms),
                     **{k: state[k] for k in ("final", "final_pids", "outcome")},
                     "first_stale": stale(first["answer"]["cited_pids"]), "final_stale": stale(state["final_pids"]),
                     "answer_ms": ms.get("answer"), "total_ms": sum(ms.values()), "fallbacks": status})
    (args.out / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    answered = [r for r in rows if not r["refused"]]
    gold_in_top = [r for r in rows if set(r["gold"]) & set(r["top5"])]
    summary = {
        "n": len(rows),
        "refused": sum(r["refused"] for r in rows) / len(rows),
        "abstained": sum(r["outcome"] != "answered" for r in rows) / len(rows),
        "refused_when_gold_in_top5": sum(r["refused"] for r in gold_in_top) / max(len(gold_in_top), 1),
        "Cite-Hit": sum(bool(set(r["cited_pids"]) & set(r["gold"])) for r in answered) / max(len(answered), 1),
        "Cite-Prec": float(np.mean([np.mean([p in r["gold"] for p in r["cited_pids"]])
                                    for r in answered if r["cited_pids"]] or [0.0])),
        "with_issue": sum(bool(r["issues"]) for r in answered) / max(len(answered), 1),
        "verified": sum(r["verified"] for r in answered) / max(len(answered), 1),
        "final": {o: sum(r["outcome"] == o for r in rows) / len(rows) for o in ("answered", "refused", "degraded")},
        "revised": sum(r["revisions"] > 0 for r in rows) / len(rows),
        "final_Cite-Hit": sum(bool(set(r["final_pids"]) & set(r["gold"])) for r in rows if r["outcome"] != "refused")
                          / max(sum(r["outcome"] != "refused" for r in rows), 1),
        "final_Cite-Hit|gold_in_top5": sum(bool(set(r["final_pids"]) & set(r["gold"])) for r in gold_in_top
                                            if r["outcome"] != "refused")
                                        / max(sum(r["outcome"] != "refused" for r in gold_in_top), 1),
        "gold_in_top5": len(gold_in_top) / len(rows),
        "fallbacks": sum(bool(r["fallbacks"]) for r in rows) / len(rows),
        "workers": args.workers,
        "answer_ms_p50": float(np.median([r["answer_ms"] for r in rows if r["answer_ms"] is not None]))
                         if args.workers == 1 else None,
        "total_ms_p50": float(np.median([r["total_ms"] for r in rows])) if args.workers == 1 else None,
        "llm": dict(llm.TOTALS),
        "cost_per_item": llm.TOTALS["cost"] / len(rows),
        "answer_model": llm.model(), "judge_model": llm.judge_model(),
        "answer_model_cost_per_item": llm.TOTALS["cost_by_model"].get(llm.model(), 0.0) / len(rows),
    }
    if args.dataset == "ecfr":
        stale_first = [r for r in answered if r["first_stale"]]
        summary["first_cites_stale"] = len(stale_first) / max(len(answered), 1)
        summary["first_cites_stale_rejected"] = sum(not r["verified"] for r in stale_first) / max(len(stale_first), 1)
        summary["final_cites_stale"] = (sum(bool(r["final_stale"]) for r in rows if r["outcome"] != "refused")
                                        / max(sum(r["outcome"] != "refused" for r in rows), 1))
    if span_pids:
        final = [r for r in rows if r["outcome"] != "refused" and r["gold_spans"]]
        summary["final_Span-Rec"] = float(np.mean([np.mean([any(p in r["final_pids"] for p in span_pids[sp])
                                                             for sp in r["gold_spans"]]) for r in final] or [0.0]))
    (args.out / f"{name}.summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
