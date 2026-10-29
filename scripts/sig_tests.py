"""Paired significance tests for the main comparisons, on the runs already on disk (no API calls).

    python scripts/sig_tests.py            # writes runs/sig_tests.md

Every comparison pairs the same turns under two systems and scores each turn 0/1. It reports both rates,
the turns where only A or only B succeeds, the exact two-sided McNemar p-value and the 95% paired
bootstrap interval of B - A (vpqa.loop). Comparisons:
- verifier: the first answer vs the final one after the version and support checks and revisions;
- DAG vs ReAct on the same 100 validation turns (and the same turns with gold documents withheld);
- the eCFR answer runs with the version filter, without it, and with stale versions injected;
- the version retrieval policies (scripts/eval_versions.py `<name>.ranked.json`);
- the stock and the fine-tuned cross-encoder on MD2D validation (scripts/train_reranker.py), when both
  `runs/retrieval_ft/*/validation.ranked.json` exist, and end to end (`runs/e2e_ft/{stock,md2d}`);
- the two answer models on the same 300 turns.

A turn succeeds when it is answered with a gold citation (answerable sets), when it is declined
(unanswerable sets: withheld turns judged truly unanswerable, near-miss rewrites), or when the answer cites
no version out of force (stale measures, scored on all turns: a declined turn cites nothing).
"""
from __future__ import annotations

import argparse
import ast
import json
import random
from pathlib import Path

from vpqa.loop import bootstrap_diff, mcnemar

MD2D = "md2d_validation_n300_s0"
MODELS = {"Sonnet": Path("runs/answers"), "DeepSeek": Path("runs/answers_dsv4flash")}
REACT = {"Sonnet": Path("runs/react"), "DeepSeek": Path("runs/react_dsv4flash")}


def load(path: Path) -> dict[str, dict]:
    rows = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if isinstance(r.get("cited_pids"), str):
            r["cited_pids"] = ast.literal_eval(r["cited_pids"])
        rows[r["id"]] = r
    return rows


def model_label(run_path: Path) -> str:
    """Read the producing model from the run summary; mark old runs without metadata."""
    summary_path = run_path.with_suffix(".summary.json")
    if summary_path.exists():
        model = json.loads(summary_path.read_text()).get("answer_model")
        if model:
            normalized = str(model).lower()
            if "deepseek" in normalized:
                return "DeepSeek V4 Flash"
            if "sonnet" in normalized:
                return "Sonnet"
            return str(model)
    return "legacy (answer_model unavailable)"


def first_hit(r: dict) -> bool:
    return not r["refused"] and bool(set(r["cited_pids"]) & set(r["gold"]))


def final_hit(r: dict) -> bool:
    return r["outcome"] == "answered" and bool(set(r["final_pids"]) & set(r["gold"]))


def first_declined(r: dict) -> bool:
    return r["refused"]


def final_declined(r: dict) -> bool:
    return r["outcome"] != "answered"


def first_clean(r: dict) -> bool:
    return r["refused"] or not r["first_stale"]


def final_clean(r: dict) -> bool:
    return r["outcome"] != "answered" or not r["final_stale"]


def row(label: str, a: list[bool], b: list[bool]) -> str:
    n = len(a)
    oa, ob, p = mcnemar(a, b)
    lo, hi = bootstrap_diff([float(x) for x in a], [float(x) for x in b])
    return f"| {label} | {n} | {sum(a) / n:.3f} | {sum(b) / n:.3f} | {oa} / {ob} | {p:.3g} | [{lo:+.3f}, {hi:+.3f}] |"


def ratio_row(label: str, ans_a, hit_a, ans_b, hit_b, samples: int = 10000, seed: int = 0) -> str:
    """Cite-Hit as reported (gold cited among answered turns): both rates and the 95% paired bootstrap
    interval of B - A, resampling turns."""
    rng, n = random.Random(seed), len(ans_a)
    rate = lambda ans, hit, idx: sum(hit[i] for i in idx) / max(1, sum(ans[i] for i in idx))
    diffs = sorted(rate(ans_b, hit_b, idx) - rate(ans_a, hit_a, idx)
                   for idx in ([rng.randrange(n) for _ in range(n)] for _ in range(samples)))
    full = range(n)
    return (f"| {label} | {n} | {rate(ans_a, hit_a, full):.3f} | {rate(ans_b, hit_b, full):.3f} | - | - | "
            f"[{diffs[int(0.025 * samples)]:+.3f}, {diffs[int(0.975 * samples) - 1]:+.3f}] |")


def pair(label: str, ra: dict, rb: dict, fa, fb, ids=None) -> str:
    ids = sorted(ra.keys() & rb.keys()) if ids is None else [i for i in ids if i in ra and i in rb]
    return row(label, [fa(ra[i]) for i in ids], [fb(rb[i]) for i in ids])


def version_rows(path: Path, gold: dict[str, str], stale: dict[str, set[str]]) -> list[str]:
    data = json.loads(path.read_text())
    qids, ranked = data["qids"], data["ranked"]
    r1 = lambda run: [bool(x) and x[0] == gold[q] for x, q in zip(run, qids)]
    clean1 = lambda run: [not (x and x[0] in stale[q]) for x, q in zip(run, qids)]
    out = []
    for m in ("rrf", "rrf+rerank"):
        base = ranked[f"unaware / {m}"]
        for policy in ("latest", "prefilter", "postfilter:20", "postfilter:100"):
            if f"{policy} / {m}" in ranked:
                other = ranked[f"{policy} / {m}"]
                out.append(row(f"{path.name.split('_')[0]} {m}: unaware → {policy}, R@1", r1(base), r1(other)))
                out.append(row(f"{path.name.split('_')[0]} {m}: unaware → {policy}, not stale@1", clean1(base), clean1(other)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--covered", type=Path, default=Path(f"runs/answers/{MD2D}_withheld.covered.json"))
    ap.add_argument("--qa", type=Path, default=Path("data/benchmark/ecfr_version_qa.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("runs/sig_tests.md"))
    args = ap.parse_args()
    covered = json.loads(args.covered.read_text())
    unanswerable = sorted(i for i, c in covered.items() if not c)

    head = ["| Comparison (A → B) | Turns | A | B | Only A / only B | McNemar p | 95% CI of B − A |",
            "|---|---|---|---|---|---|---|"]
    lines = ["Paired tests on frozen runs (scripts/sig_tests.py).", ""]

    lines += ["## Verifier: first answer → final", ""] + head
    for model, d in MODELS.items():
        ans, wh, nm = (load(d / f"{MD2D}{s}.jsonl") for s in ("", "_withheld", "_nearmiss"))
        lines.append(pair(f"{model}, answerable: answered with gold citation", ans, ans, first_hit, final_hit))
        lines.append(pair(f"{model}, withheld: declined", wh, wh, first_declined, final_declined, unanswerable))
        lines.append(pair(f"{model}, near-miss: declined", nm, nm, first_declined, final_declined))
    for name in ("ecfr_dev_n100_s0", "ecfr_dev_n100_s0_unaware", "ecfr_test_n284_s0_unaware", "ecfr_test_n284_s0_inject"):
        path = MODELS["Sonnet"] / f"{name}.jsonl"
        rows = load(path)
        model = model_label(path)
        lines.append(pair(f"{model}, {name}: cites no stale version", rows, rows, first_clean, final_clean))
        lines.append(pair(f"{model}, {name}: answered with gold citation", rows, rows, first_hit, final_hit))

    lines += ["", "## DAG (final) → ReAct, same turns", ""] + head
    for model in MODELS:
        dag, react = load(MODELS[model] / f"{MD2D}.jsonl"), load(REACT[model] / "md2d_validation_n100_s0.jsonl")
        lines.append(pair(f"{model}, answerable: answered with gold citation", dag, react, final_hit, first_hit))
        lines.append(pair(f"{model}, answerable: declined", dag, react, final_declined, first_declined))
        ids = sorted(dag.keys() & react.keys())
        lines.append(ratio_row(f"{model}, answerable: Cite-Hit among answered",
                               [not final_declined(dag[i]) for i in ids], [final_hit(dag[i]) for i in ids],
                               [not first_declined(react[i]) for i in ids], [first_hit(react[i]) for i in ids]))
        dag, react = (load(MODELS[model] / f"{MD2D}_withheld.jsonl"),
                      load(REACT[model] / "md2d_validation_n100_s0_withheld.jsonl"))
        lines.append(pair(f"{model}, withheld: declined", dag, react, final_declined, first_declined, unanswerable))

    lines += ["", "## eCFR answers: version filter comparisons (final)", ""] + head
    for split in ("dev_n100", "test_n284"):
        base_path = MODELS["Sonnet"] / f"ecfr_{split}_s0.jsonl"
        base = load(base_path)
        for variant in ("unaware", "inject"):
            path = MODELS["Sonnet"] / f"ecfr_{split}_s0_{variant}.jsonl"
            if path.exists():
                other = load(path)
                models = f"filter [{model_label(base_path)}] → {variant} [{model_label(path)}]"
                lines.append(pair(f"{split}: {models}, answered with gold citation", base, other, final_hit, final_hit))
                lines.append(pair(f"{split}: {models}, first answer cites no stale version",
                                  base, other, first_clean, first_clean))

    lines += ["", "## Version retrieval: unaware → policy", ""] + head
    items = [json.loads(l) for l in args.qa.read_text().splitlines()]
    gold = {it["qid"]: it["gold_pid"] for it in items}
    stale = {it["qid"]: set(it["other_versions"]) for it in items}
    for path in sorted(Path("runs/versions").glob("*.ranked.json")):
        lines += version_rows(path, gold, stale)

    stock, tuned = Path("runs/retrieval_ft/stock/validation.ranked.json"), Path("runs/retrieval_ft/md2d/validation.ranked.json")
    if stock.exists() and tuned.exists():
        lines += ["", "## MD2D retrieval: stock → fine-tuned cross-encoder (router, gated query)", ""] + head
        a, b = json.loads(stock.read_text()), json.loads(tuned.read_text())
        assert a["ids"] == b["ids"]
        hit = lambda run, k: [bool(set(x[:k]) & set(g)) for x, g in zip(run, a["gold"])]
        for n in (10, 20, 40):
            name = f"rrf+rerank@{n}/gated+route:router"
            for k in (1, 5):
                lines.append(row(f"rerank@{n}, Hit@{k}", hit(a["ranked"][name], k), hit(b["ranked"][name], k)))

    for e2e, run in ((Path("runs/e2e_ft"), MD2D), (Path("runs/e2e_ft_train"), "md2d_train_n200_s1")):
        if not (e2e / "md2d" / f"{run}.jsonl").exists():
            continue
        lines += ["", f"## End to end, {run}: stock → fine-tuned cross-encoder (DeepSeek, 8 workers, final)", ""] + head
        a, b = (load(e2e / s / f"{run}.jsonl") for s in ("stock", "md2d"))
        lines.append(pair("answerable: answered with gold citation", a, b, final_hit, final_hit))
        lines.append(pair("answerable: declined", a, b, final_declined, final_declined))
        # a better reranker can surface another document that covers the withheld one, so a withheld turn
        # counts as unanswerable here only when the judge finds neither run's own top 5 covers it
        cov = [e2e / s / f"{run}_withheld.covered.json" for s in ("stock", "md2d")]
        if all(p.exists() for p in cov):
            ca, cb = (json.loads(p.read_text()) for p in cov)
            both = sorted(i for i in ca if not ca[i] and not cb.get(i, True))
            lines.append(pair("withheld, judged covered by neither top 5: declined",
                              *(load(e2e / s / f"{run}_withheld.jsonl") for s in ("stock", "md2d")),
                              final_declined, final_declined, both))
        for suffix, label, ids in (("_withheld", "withheld (original judgment)", unanswerable if run == MD2D else []),
                                   ("_nearmiss", "near-miss", None)):
            if ids != [] and (e2e / "md2d" / f"{run}{suffix}.jsonl").exists():
                a, b = (load(e2e / s / f"{run}{suffix}.jsonl") for s in ("stock", "md2d"))
                lines.append(pair(f"{label}: declined", a, b, final_declined, final_declined, ids))

    lines += ["", "## Answer model: Sonnet → DeepSeek, same turns (final)", ""] + head
    s, k = (load(MODELS[m] / f"{MD2D}.jsonl") for m in MODELS)
    lines.append(pair("answerable: answered with gold citation", s, k, final_hit, final_hit))
    lines.append(pair("answerable: declined", s, k, final_declined, final_declined))
    s, k = (load(MODELS[m] / f"{MD2D}_withheld.jsonl") for m in MODELS)
    lines.append(pair("withheld: declined", s, k, final_declined, final_declined, unanswerable))
    s, k = (load(MODELS[m] / f"{MD2D}_nearmiss.jsonl") for m in MODELS)
    lines.append(pair("near-miss: declined", s, k, final_declined, final_declined))

    report = "\n".join(lines)
    args.out.write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
