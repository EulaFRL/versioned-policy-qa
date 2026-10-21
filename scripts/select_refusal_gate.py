"""Pick the rerank-score refusal threshold on train, then score refusal on the paired validation runs.

    python scripts/select_refusal_gate.py --train-n 1000
    python scripts/select_refusal_gate.py --report    # reuse saved scores
    python scripts/select_refusal_gate.py --report --covered   # drop withheld turns other documents still answer

A turn is refused without an LLM call when the best rerank score is below the threshold (configs
`answer.min_score`). Train: a fixed-seed sample of answerable turns and the same turns with their gold
documents withheld (vpqa.eval.refusal); the threshold maximises refusal F1 of the gate alone. Validation:
the 300 turns of scripts/eval_answers.py (same seed) and their withheld versions. The report combines the
gate with the answers of both eval_answers.py runs:

- refusal P / R / F1 (unanswerable = withheld) for the gate alone, the first answer (no verifier), the
  final output (verifier and revisions; "abstain" also counts passage text shown instead of an answer),
  and the gate followed by the final output;
- the coverage-accuracy curve of gate + final output over thresholds: coverage is the share of all
  turns answered, accuracy the share of answered turns that are answerable and cite a gold passage.

With --covered, validation pairs whose withheld turn is still answerable from other documents
(scripts/judge_withheld.py) are dropped; the train threshold is unchanged.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import yaml

from vpqa.data.multidoc2dial import load_passages, load_turns
from vpqa.eval.metrics import refusal_prf
from vpqa.eval.refusal import best_threshold, coverage_accuracy, withhold_gold_docs
from vpqa.pipeline.dag import Pipeline

RAW = Path("data/raw/multidoc2dial")
DOC = RAW / "multidoc2dial_doc.json"


def sample(split: str, n: int, seed: int):
    turns = [t for t in load_turns(RAW / f"multidoc2dial_dial_{split}.json", DOC) if t.answerable]
    return random.Random(seed).sample(turns, min(n, len(turns)))


def scores(pipe: Pipeline, turns) -> dict[str, float]:
    out = {}
    for t in turns:
        state, _ = pipe.run(t)
        out[f"{t.dialog_id}/{t.turn_id}"] = state["top_score"]
    return out


def load_rows(path: Path) -> dict[str, dict]:
    return {r["id"]: r for r in map(json.loads, path.read_text().splitlines())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-n", type=int, default=1000)
    ap.add_argument("--val-n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--router", type=Path, default=Path("runs/router/router.pkl"))
    ap.add_argument("--answers", type=Path, default=Path("runs/answers"))
    ap.add_argument("--out", type=Path, default=Path("runs/refusal"))
    ap.add_argument("--report", action="store_true", help="skip retrieval, reuse saved scores")
    ap.add_argument("--covered", action="store_true", help="drop pairs judged still answerable when withheld")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"scores_train{args.train_n}_val{args.val_n}_s{args.seed}.json"

    if not args.report:
        cfg = yaml.safe_load(open(args.config))
        pipe = Pipeline(load_passages(DOC), cfg, Path("data/index/BAAI__bge-m3.npy"), args.router, timeouts=False)
        saved = {}
        for split, n in (("train", args.train_n), ("validation", args.val_n)):
            turns = sample(split, n, args.seed)
            saved[split] = {"answerable": scores(pipe, turns),
                            "withheld": scores(pipe, withhold_gold_docs(turns, pipe.by_pid))}
        path.write_text(json.dumps(saved))
    saved = json.loads(path.read_text())

    tr = saved["train"]
    ids = sorted(tr["answerable"])
    t_scores = [tr["answerable"][i] for i in ids] + [tr["withheld"][i] for i in ids]
    threshold, f1 = best_threshold(t_scores, [False] * len(ids) + [True] * len(ids))
    print(f"train: {len(ids)} pairs, threshold {threshold:.3f} (gate-alone refusal F1 {f1:.3f})")

    va = saved["validation"]
    name = f"md2d_validation_n{args.val_n}_s{args.seed}"
    ans, wh = load_rows(args.answers / f"{name}.jsonl"), load_rows(args.answers / f"{name}_withheld.jsonl")
    ids = sorted(ans.keys() & wh.keys() & va["answerable"].keys())
    if args.covered:
        covered = json.loads((args.answers / f"{name}_withheld.covered.json").read_text())
        ids = [i for i in ids if not covered[i]]
    rows = [(ans[i], va["answerable"][i], False) for i in ids] + [(wh[i], va["withheld"][i], True) for i in ids]
    unans = [u for _, _, u in rows]
    gate = lambda t: [s is not None and s < t for _, s, _ in rows]
    readings = {
        "gate alone": gate(threshold),
        "first answer (no verifier)": [r["refused"] for r, _, _ in rows],
        "final, strict": [r["outcome"] == "refused" for r, _, _ in rows],
        "final, abstain": [r["outcome"] != "answered" for r, _, _ in rows],
        "gate + final, strict": [g or r["outcome"] == "refused" for g, (r, _, _) in zip(gate(threshold), rows)],
        "gate + final, abstain": [g or r["outcome"] != "answered" for g, (r, _, _) in zip(gate(threshold), rows)],
    }
    lines = [f"validation: {len(ids)} pairs; threshold {threshold:.3f} picked on {len(tr['answerable'])} train pairs",
             "", "| Refusal | P | R | F1 |", "|---|---|---|---|"]
    for k, pred in readings.items():
        p, r, f = refusal_prf(pred, unans)
        lines.append(f"| {k} | {p:.3f} | {r:.3f} | {f:.3f} |")

    correct = [not u and bool(set(r["final_pids"]) & set(r["gold"])) for r, _, u in rows]
    lines += ["", "| Threshold | Coverage | Accuracy | Refusal F1 |", "|---|---|---|---|"]
    all_scores = [s for _, s, _ in rows if s is not None]
    for t in [None, *np.quantile(all_scores, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).tolist(), threshold]:
        g = gate(t) if t is not None else [False] * len(rows)
        answered = [not x and r["outcome"] == "answered" for x, (r, _, _) in zip(g, rows)]
        cov, acc = coverage_accuracy(answered, correct)
        f = refusal_prf([not a for a in answered], unans)[2]
        label = "none" if t is None else f"{t:.3f}" + (" (train)" if t == threshold else "")
        lines.append(f"| {label} | {cov:.3f} | {acc:.3f} | {f:.3f} |")
    report = "\n".join(lines)
    (args.out / f"{name}_refusal{'_uncovered' if args.covered else ''}.md").write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
