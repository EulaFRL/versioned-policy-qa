"""Offline improvement loop on frozen MultiDoc2Dial turns (vpqa.loop).

    python scripts/loop.py                 # rounds on train, then one confirmation on validation
    python scripts/loop.py --no-confirm

Frozen sets: the answerable turns of scripts/eval_answers.py (`--split`, `--n`, `--seed`) and their
near-miss rewrites (scripts/make_near_miss.py). Each round:

1. run the current configuration on both sets and attribute every failure to a category;
2. take the category with the most failures that still has untried patches (configs/skills.yaml);
3. run the current configuration plus each of its patches on the same turns. The patch's target set is
   the near-miss set for knowledge gaps and the answerable set otherwise; the other set is its guard.
   A patch passes if success on the target set rises with an exact McNemar p below `--alpha` and success
   on the guard set does not fall with p below `--alpha`;
4. keep the passing patch with the largest gain and make it the current configuration; if none passes,
   the configuration stays as it was (rollback) and the category is marked tried.

The loop ends when no category with failures has an untried patch. The final configuration is then run
once on the validation turns (`--confirm-n`, seed 0, and their near-miss rewrites) against the starting one.
Every run, decision and count goes to <out>/ledger.jsonl and <out>/report.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import yaml

from vpqa.loop import CATEGORIES, attribute, bootstrap_diff, mcnemar, merge, read_trace, success


def evaluate(cfg: dict, split: str, n: int, seed: int, out: Path, workers: int) -> dict:
    """Run eval_answers.py with `cfg` on the answerable and near-miss sets; rows and traces by set."""
    key = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10]
    run_dir = out / "runs" / key
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    name = f"md2d_{split}_n{n}_s{seed}"
    result = {"dir": str(run_dir)}
    for label, withhold, suffix in (("answerable", "none", ""), ("near-miss", "near-miss", "_nearmiss")):
        rows_path = run_dir / f"{name}{suffix}.jsonl"
        if not rows_path.exists():
            subprocess.run([sys.executable, "scripts/eval_answers.py", "--split", split, "--n", str(n), "--seed", str(seed),
                            "--withhold", withhold, "--config", str(run_dir / "config.yaml"), "--out", str(run_dir),
                            "--workers", str(workers)], check=True, stdout=subprocess.DEVNULL, env=os.environ)
        rows = {r["id"]: r for r in map(json.loads, rows_path.read_text().splitlines())}
        result[label] = (rows, read_trace(run_dir / f"{name}{suffix}.trace.jsonl"))
    return result


def outcomes(res: dict, top_n: int) -> dict:
    """Per set: success by turn id; failure categories."""
    out = {}
    for label in ("answerable", "near-miss"):
        rows, traces = res[label]
        ok = label == "answerable"
        out[label] = {i: success(r, ok) for i, r in rows.items()}
        out[label + " categories"] = Counter(c for i, r in rows.items() if (c := attribute(r, traces[i], ok, top_n)))
    return out


def compare(a: dict, b: dict, label: str) -> dict:
    ids = sorted(a[label].keys() & b[label].keys())
    xa, xb = [a[label][i] for i in ids], [b[label][i] for i in ids]
    only_a, only_b, p = mcnemar(xa, xb)
    lo, hi = bootstrap_diff(xa, xb)
    return {"n": len(ids), "a": sum(xa) / len(ids), "b": sum(xb) / len(ids), "only_a": only_a, "only_b": only_b,
            "p": p, "ci95": [lo, hi]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--skills", default="configs/skills.yaml")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-confirm", action="store_true")
    ap.add_argument("--confirm-n", type=int, default=300)
    ap.add_argument("--out", type=Path, default=Path("runs/loop"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    ledger_path = args.out / "ledger.jsonl"
    ledger_path.unlink(missing_ok=True)

    def log(entry: dict) -> None:
        with ledger_path.open("a") as f:
            f.write(json.dumps(entry) + "\n")
        print(json.dumps(entry), flush=True)

    start = yaml.safe_load(open(args.config))
    skills = yaml.safe_load(open(args.skills))
    current, applied, tried = start, [], set()
    top_n = lambda cfg: cfg["retrieval"]["top_n"]
    report = [f"# Offline loop on {args.split} (n {args.n}, seed {args.seed}; alpha {args.alpha})", ""]
    for rnd in range(1, len(CATEGORIES) + 1):
        base = outcomes(evaluate(current, args.split, args.n, args.seed, args.out, args.workers), top_n(current))
        failures = base["answerable categories"] + base["near-miss categories"]
        log({"round": rnd, "applied": applied, "failures": dict(failures),
             "success": {k: sum(base[k].values()) / len(base[k]) for k in ("answerable", "near-miss")}})
        report += [f"## Round {rnd}: applied {applied or 'none'}", "",
                   "Failures: " + ", ".join(f"{c} {failures[c]}" for c in CATEGORIES), ""]
        open_cats = [c for c, _ in failures.most_common() if c not in tried and any(s["category"] == c for s in skills)]
        if not open_cats:
            report += ["No category with failures has an untried patch; the loop ends.", ""]
            break
        cat = open_cats[0]
        tried.add(cat)
        target, guard = ("near-miss", "answerable") if cat == "knowledge gap" else ("answerable", "near-miss")
        report += [f"Category: **{cat}** (target set {target}, guard set {guard})", "",
                   "| Patch | Target before → after | Only before / only after | p | 95% CI of gain | Guard before → after | p | Decision |",
                   "|---|---|---|---|---|---|---|---|"]
        passed = []
        for skill in [s for s in skills if s["category"] == cat]:
            cfg = merge(current, skill["patch"])
            res = outcomes(evaluate(cfg, args.split, args.n, args.seed, args.out, args.workers), top_n(cfg))
            t, g = compare(base, res, target), compare(base, res, guard)
            ok = t["only_b"] > t["only_a"] and t["p"] < args.alpha and not (g["only_a"] > g["only_b"] and g["p"] < args.alpha)
            log({"round": rnd, "category": cat, "skill": skill["name"], "patch": skill["patch"], "target": t, "guard": g,
                 "passed": ok, "failures_after": dict(res["answerable categories"] + res["near-miss categories"])})
            report.append(f"| {skill['name']} | {t['a']:.3f} → {t['b']:.3f} | {t['only_a']} / {t['only_b']} | {t['p']:.3g} | "
                          f"[{t['ci95'][0]:+.3f}, {t['ci95'][1]:+.3f}] | {g['a']:.3f} → {g['b']:.3f} | {g['p']:.3g} | "
                          f"{'pass' if ok else 'reject'} |")
            if ok:
                passed.append((t["b"] - t["a"], skill))
        report.append("")
        if passed:
            _, best = max(passed, key=lambda x: x[0])
            current, applied = merge(current, best["patch"]), applied + [best["name"]]
            report += [f"Kept **{best['name']}**.", ""]
            log({"round": rnd, "kept": best["name"]})
        else:
            report += ["No patch passed; configuration rolled back to the round's start.", ""]
            log({"round": rnd, "kept": None})

    (args.out / "final.yaml").write_text(yaml.safe_dump(current, sort_keys=False))
    report += [f"Final configuration: {applied or 'unchanged'} ({args.out / 'final.yaml'}).", ""]
    if not args.no_confirm and applied:
        a = outcomes(evaluate(start, "validation", args.confirm_n, 0, args.out, args.workers), top_n(start))
        b = outcomes(evaluate(current, "validation", args.confirm_n, 0, args.out, args.workers), top_n(current))
        report += [f"## Confirmation on validation (n {args.confirm_n}, seed 0)", "",
                   "| Set | Turns | Start | Final | Only start / only final | p | 95% CI of gain |", "|---|---|---|---|---|---|---|"]
        for label in ("answerable", "near-miss"):
            c = compare(a, b, label)
            report.append(f"| {label} | {c['n']} | {c['a']:.3f} | {c['b']:.3f} | {c['only_a']} / {c['only_b']} | {c['p']:.3g} | "
                          f"[{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}] |")
            log({"confirm": label, **c})
    (args.out / "report.md").write_text("\n".join(report) + "\n")
    print("\n".join(report))


if __name__ == "__main__":
    main()
