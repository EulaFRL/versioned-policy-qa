"""ReAct control group on the turns of scripts/eval_answers.py, compared with the fixed DAG.

    python scripts/eval_react.py --n 100
    python scripts/eval_react.py --n 100 --withhold gold-doc
    python scripts/eval_react.py --dataset ecfr --n 100

Same seed, so the turns are the first n of the eval_answers.py sample (and of its withheld run); on eCFR
the same n dev questions as eval_answers.py, with the as-of date stated in the user message and an as_of
argument on the search tool. Same LLM and retrieval (vpqa.pipeline.react). After the run, each ReAct answer goes through the DAG's verifier once,
only to measure how often it is supported; nothing is revised.

Reported for ReAct: searches per turn, share of turns answered without any search, share of turns that
repeat a search (a query with the same words as an earlier one in that turn), share that hit the search
budget; on eCFR, the share of searches that set the as-of date and set it right, and the share of answers
citing a version not in force on that date; refusal, Cite-Hit, Cite-Prec, parse issues, the verifier's pass rate, latency and cost. The
comparison table puts ReAct next to the DAG's first answer (before the verifier) and final output on the
same turns; on withheld turns every answer is wrong, so the share answered is the error rate.
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
from vpqa.eval.refusal import withhold_gold_docs
from vpqa.pipeline import verifier
from vpqa.pipeline.dag import Pipeline
from vpqa.pipeline.react import ReactAgent, timed
from vpqa.retrieval.bm25 import tokenize
from vpqa.schema import Turn

RAW = Path("data/raw/multidoc2dial")
DOC = RAW / "multidoc2dial_doc.json"


def repeats(queries: list[str]) -> bool:
    seen = []
    for q in queries:
        words = frozenset(tokenize(q))
        if words in seen:
            return True
        seen.append(words)
    return False


def rate(rows: list[dict], pred) -> float:
    return sum(map(pred, rows)) / max(len(rows), 1)


def scores(rows: list[dict], answered_key: str, pids_key: str) -> dict:
    answered = [r for r in rows if r[answered_key]]
    return {
        "answered": len(answered) / max(len(rows), 1),
        "Cite-Hit": rate(answered, lambda r: bool(set(r[pids_key]) & set(r["gold"]))),
        "Cite-Prec": float(np.mean([np.mean([p in r["gold"] for p in r[pids_key]])
                                    for r in answered if r[pids_key]] or [0.0])),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["md2d", "ecfr"], default="md2d")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--sample-n", type=int, default=300, help="size of the eval_answers.py sample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--withhold", choices=["none", "gold-doc"], default="none")
    ap.add_argument("--max-searches", type=int, default=4)
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--router", type=Path, default=Path("runs/router/router.pkl"))
    ap.add_argument("--qa", type=Path, default=Path("data/benchmark/ecfr_version_qa.jsonl"))
    ap.add_argument("--answers", type=Path, default=Path("runs/answers"))
    ap.add_argument("--out", type=Path, default=Path("runs/react"))
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    a = cfg.get("answer", {})

    if args.dataset == "md2d":
        turns = [t for t in load_turns(RAW / "multidoc2dial_dial_validation.json", DOC) if t.answerable]
        turns = random.Random(args.seed).sample(turns, min(args.sample_n, len(turns)))[:args.n]
        pipe = Pipeline(load_passages(DOC), cfg, Path("data/index/BAAI__bge-m3.npy"), args.router, timeouts=False)
        name = f"md2d_validation_n{args.n}_s{args.seed}"
        dag_name = f"md2d_validation_n{args.sample_n}_s{args.seed}"
    else:
        items = [json.loads(l) for l in args.qa.read_text().splitlines()]
        turns = [Turn(it["qid"], 0, "", [], it["question"], [it["gold_pid"]], as_of=date.fromisoformat(it["as_of"]))
                 for it in items if it["split"] == "dev"]
        turns = random.Random(args.seed).sample(turns, min(args.n, len(turns)))
        pipe = Pipeline(load_versions(), cfg, Path("data/index/BAAI__bge-m3__ecfr.npy"), timeouts=False)
        name = dag_name = f"ecfr_dev_n{args.n}_s{args.seed}"
    if args.withhold == "gold-doc":
        turns = withhold_gold_docs(turns, pipe.by_pid)
    agent = ReactAgent(pipe, llm.chat_tools, args.max_searches, a.get("history_turns", 4),
                       a.get("passage_words", 400), cfg["retrieval"]["top_n"])

    suffix = "_withheld" if args.withhold != "none" else ""
    stale = lambda t, pids: [p for p in pids if not verifier.in_force(pipe.by_pid[p], t.as_of)]
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for t in turns:
        r, ms = timed(agent, t)
        cost_before = llm.TOTALS["cost"]
        supported = None
        if not r.refused:
            v = verifier.verify(r.claims, r.issues, pipe.by_pid, t.as_of, llm.judge_json, a.get("passage_words", 400))
            supported = v.passed
        rows.append({"id": f"{t.dialog_id}/{t.turn_id}", "query": t.query, "gold": t.gold_pids,
                     "answer": r.answer, "cited_pids": r.cited_pids, "refused": r.refused, "issues": r.issues,
                     "as_of": str(t.as_of) if t.as_of else None, "stale": stale(t, r.cited_pids),
                     "searches": r.searches, "llm_calls": r.llm_calls, "forced": r.forced, "supported": supported,
                     "judge_cost": llm.TOTALS["cost"] - cost_before, "ms": ms})
    name += suffix
    (args.out / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    for r in rows:
        r["react_answered"] = not r["refused"]
    answered = [r for r in rows if not r["refused"]]
    judge_cost = sum(r["judge_cost"] for r in rows)
    summary = {
        "n": len(rows),
        "searches_mean": float(np.mean([len(r["searches"]) for r in rows])),
        "no_search": rate(rows, lambda r: not r["searches"]),
        "no_search_answered": rate(rows, lambda r: not r["searches"] and not r["refused"]),
        "repeated_search": rate(rows, lambda r: repeats([s["query"] for s in r["searches"]])),
        "department_set": rate([s for r in rows for s in r["searches"]], lambda s: s["department"] is not None),
        "hit_budget": rate(rows, lambda r: r["forced"]),
        "refused": rate(rows, lambda r: r["refused"]),
        **{k: v for k, v in scores(rows, "react_answered", "cited_pids").items() if k != "answered"},
        "with_issue": rate(answered, lambda r: bool(r["issues"])),
        "supported": rate(answered, lambda r: r["supported"]),
        "llm_calls_mean": float(np.mean([r["llm_calls"] for r in rows])),
        "ms_p50": float(np.median([r["ms"] for r in rows])),
        "ms_p95": float(np.percentile([r["ms"] for r in rows], 95)),
        "model": llm.model(), "judge_model": llm.judge_model(),
        "cost_per_item": (llm.TOTALS["cost"] - judge_cost) / len(rows),
        "judge_cost_per_item": judge_cost / len(rows),
    }
    if args.dataset == "ecfr":
        found = [s for r in rows for s in r["searches"]]
        summary["as_of_set"] = rate(found, lambda s: bool(s["as_of"]))
        summary["as_of_right"] = rate([(s, r) for r in rows for s in r["searches"]], lambda x: x[0]["as_of"] == x[1]["as_of"])
        summary["cites_stale"] = rate(answered, lambda r: bool(r["stale"]))
    (args.out / f"{name}.summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    dag_path = args.answers / f"{dag_name}{suffix}.jsonl"
    if dag_path.exists():
        dag = {r["id"]: r for r in map(json.loads, dag_path.read_text().splitlines())}
        ids = [r["id"] for r in rows if r["id"] in dag]
        d = [dag[i] for i in ids]
        for r in d:
            r["first_answered"], r["final_answered"] = not r["refused"], r["outcome"] == "answered"
        react = [r for r in rows if r["id"] in dag]
        table = {"DAG, first answer": {**scores(d, "first_answered", "cited_pids"),
                                       "supported": rate([r for r in d if r["first_answered"]], lambda r: r["verified"]),
                                       "searches": 1.0},
                 "DAG, final": {**scores(d, "final_answered", "final_pids"), "supported": None, "searches": 1.0},
                 "ReAct": {**scores(react, "react_answered", "cited_pids"), "supported": summary["supported"],
                           "searches": summary["searches_mean"]}}
        if args.dataset == "ecfr":  # share of answers citing a version not in force on the as-of date
            for (k, key, ans) in (("DAG, first answer", "first_stale", "first_answered"),
                                  ("DAG, final", "final_stale", "final_answered"), ("ReAct", "stale", "react_answered")):
                rows_k = react if k == "ReAct" else d
                table[k]["stale"] = rate([r for r in rows_k if r[ans]], lambda r: bool(r[key]))
        stale_col = args.dataset == "ecfr"
        lines = [f"{len(ids)} turns{' (gold documents withheld)' if suffix else ''}", "",
                 "| System | Answered | Cite-Hit | Cite-Prec | Verifier pass | Searches |" + (" Cites stale |" if stale_col else ""),
                 "|---|---|---|---|---|---|" + ("---|" if stale_col else "")]
        for k, m in table.items():
            sup = "-" if m["supported"] is None else f"{m['supported']:.3f}"
            lines.append(f"| {k} | {m['answered']:.3f} | {m['Cite-Hit']:.3f} | {m['Cite-Prec']:.3f} | {sup} "
                         f"| {m['searches']:.2f} |" + (f" {m['stale']:.3f} |" if stale_col else ""))
        report = "\n".join(lines)
        (args.out / f"{name}.compare.md").write_text(report + "\n")
        print(report)


if __name__ == "__main__":
    main()
