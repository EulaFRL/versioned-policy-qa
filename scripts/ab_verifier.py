"""A/B of the verifier's question check (scripts/eval_answers.py --check-question) on frozen paired sets.

    python scripts/ab_verifier.py runs/answers_dsv4flash

For one answer model, the runs without (A) and with (B) the check on the same turns: the 300 answerable
validation turns, the same turns with their gold documents withheld (only those judged truly unanswerable,
scripts/judge_withheld.py), and their near-miss rewrites (scripts/make_near_miss.py). The first answers are
identical in A and B (same prompts, cached replies); only the verdicts and what follows them differ.

Per set: the share of turns the system declines (final reply NOT_FOUND or passage text shown instead of an
answer), and on answerable turns the share answered with a gold citation. Paired differences are tested
with an exact McNemar test on the turns where A and B disagree.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from vpqa.eval.metrics import refusal_prf
from vpqa.loop import mcnemar

NAME = "md2d_validation_n300_s0"


def load(path: Path) -> dict[str, dict]:
    return {r["id"]: r for r in map(json.loads, path.read_text().splitlines())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", type=Path)
    ap.add_argument("--covered", type=Path, default=Path(f"runs/answers/{NAME}_withheld.covered.json"))
    args = ap.parse_args()
    covered = json.loads(args.covered.read_text())
    sets = {"answerable": "", "withheld": "_withheld", "near-miss": "_nearmiss"}
    a = {k: load(args.runs / f"{NAME}{s}.jsonl") for k, s in sets.items()}
    b = {k: load(args.runs / f"{NAME}{s}_cq.jsonl") for k, s in sets.items()}
    ids = {k: sorted(a[k].keys() & b[k].keys()) for k in sets}
    ids["withheld"] = [i for i in ids["withheld"] if not covered[i]]

    declined = lambda r: r["outcome"] != "answered"
    hit = lambda r: r["outcome"] == "answered" and bool(set(r["final_pids"]) & set(r["gold"]))
    lines = [f"Runs: {args.runs}", "", "| Set | Turns | Measure | Without check | With check | Only without / only with | McNemar p |",
             "|---|---|---|---|---|---|---|"]
    measures = {"answerable": [("declined (false refusal)", declined), ("answered with gold citation", hit)],
                "withheld": [("declined", declined)], "near-miss": [("declined", declined)]}
    for k, ms in measures.items():
        for label, f in ms:
            xa, xb = [f(a[k][i]) for i in ids[k]], [f(b[k][i]) for i in ids[k]]
            oa, ob, p = mcnemar(xa, xb)
            n = len(ids[k])
            lines.append(f"| {k} | {n} | {label} | {sum(xa) / n:.3f} | {sum(xb) / n:.3f} | {oa} / {ob} | {p:.3g} |")

    lines += ["", "Refusal P / R / F1 (declined = final NOT_FOUND or passage text), answerable turns paired with each "
              "unanswerable set:", "", "| Unanswerable set | Pairs | Without check | With check |", "|---|---|---|---|"]
    for k in ("withheld", "near-miss"):
        pair = [i for i in ids[k] if i in a["answerable"] and i in b["answerable"]]
        cells = []
        for run in (a, b):
            pred = [declined(run["answerable"][i]) for i in pair] + [declined(run[k][i]) for i in pair]
            cells.append("{:.3f} / {:.3f} / {:.3f}".format(*refusal_prf(pred, [False] * len(pair) + [True] * len(pair))))
        lines.append(f"| {k} | {len(pair)} | {cells[0]} | {cells[1]} |")
    report = "\n".join(lines)
    (args.runs / "ab_check_question.md").write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
