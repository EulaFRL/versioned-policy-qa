"""Agreement between coverage judges from different model families (scripts/judge_withheld.py), and a blind
sample for a human check.

    python scripts/judge_agreement.py runs/judges/withheld_covered__*.json [--sample 30]

Prints Cohen's kappa for every pair of judges and Fleiss' kappa over all of them, and how the set of truly
unanswerable withheld turns (not covered) changes under a majority vote. With --sample, writes
runs/judges/human_check.md (turns to label by hand, no verdicts shown) and human_check.key.json (the
judges' verdicts): two thirds turns the judges disagree on, one third turns they all agree on, shuffled.
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path

from vpqa.data.multidoc2dial import load_passages, load_turns

RAW = Path("data/raw/multidoc2dial")
DOC = RAW / "multidoc2dial_doc.json"
RUN = Path("runs/answers/md2d_validation_n300_s0_withheld.jsonl")


def cohen(a: list[bool], b: list[bool]) -> float:
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def fleiss(labels: list[list[bool]]) -> float:
    """labels[i] = the judges' verdicts on item i."""
    n, k = len(labels), len(labels[0])
    p_true = sum(sum(l) for l in labels) / (n * k)
    pe = p_true ** 2 + (1 - p_true) ** 2
    pi = [(sum(l) * (sum(l) - 1) + (k - sum(l)) * (k - sum(l) - 1)) / (k * (k - 1)) for l in labels]
    return (sum(pi) / n - pe) / (1 - pe)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("judges", nargs="+", type=Path)
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    names = [p.stem.split("__", 1)[1] for p in args.judges]
    verdicts = {n: json.loads(p.read_text()) for n, p in zip(names, args.judges)}
    reasons = {n: json.loads(p.with_suffix(".reasons.json").read_text()) for n, p in zip(names, args.judges)}
    ids = sorted(set.intersection(*(set(v) for v in verdicts.values())))
    for n in names:  # a reply with no JSON was scored not covered; say how often
        failed = sum(reasons[n].get(i) is None for i in ids)
        print(f"{n:24} covered {sum(verdicts[n][i] for i in ids):3d} / {len(ids)}   no verdict parsed: {failed}")
    print()
    for a, b in itertools.combinations(names, 2):
        x, y = [verdicts[a][i] for i in ids], [verdicts[b][i] for i in ids]
        agree = sum(p == q for p, q in zip(x, y)) / len(ids)
        print(f"Cohen kappa  {a} vs {b}: {cohen(x, y):.3f}   (raw agreement {agree:.3f})")
    labels = [[verdicts[n][i] for n in names] for i in ids]
    if len(names) > 2:
        print(f"Fleiss kappa over {len(names)} judges: {fleiss(labels):.3f}")
        majority = {i: sum(l) * 2 > len(l) for i, l in zip(ids, labels)}
        unanimous = sum(len(set(l)) == 1 for l in labels)
        print(f"unanimous: {unanimous} / {len(ids)}")
        for n in names:
            print(f"not covered by {n:22} {sum(not verdicts[n][i] for i in ids)}")
        print(f"not covered by majority               {sum(not c for c in majority.values())}")
        out = args.judges[0].parent / "withheld_covered__majority.json"
        out.write_text(json.dumps(majority, indent=1))
        print(f"wrote {out}")
    if args.sample:
        rng = random.Random(args.seed)
        split = [i for i, l in zip(ids, labels) if len(set(l)) > 1]
        agreed = [i for i in ids if i not in set(split)]
        rng.shuffle(split)
        rng.shuffle(agreed)
        n_split = min(len(split), args.sample * 2 // 3)  # a third agreed, to check the labels nobody disputes
        picked = split[:n_split] + agreed[:args.sample - n_split]
        rng.shuffle(picked)  # the reader cannot tell disagreements by position
        write_sample(picked, names, verdicts, reasons, args.judges[0].parent)


def write_sample(picked, names, verdicts, reasons, out_dir: Path) -> None:
    rows = {r["id"]: r for r in map(json.loads, RUN.read_text().splitlines())}
    docs = json.loads(DOC.read_text())["doc_data"]
    spans = {f"{doc_id}::{sid}": sp["text_sp"].strip()
             for dom in docs.values() for doc_id, doc in dom.items() for sid, sp in doc["spans"].items()}
    by_pid = {p.pid: p for p in load_passages(DOC)}
    turns = {f"{t.dialog_id}/{t.turn_id}": t for t in load_turns(RAW / "multidoc2dial_dial_validation.json", DOC)}
    lines = ["# Withheld turns: are they still answerable?", "",
             "For each item: do the PASSAGES contain the information from the GOLD text that the reply needs?",
             "Write covered: yes / no under each item. The judges' verdicts are in human_check.key.json.", ""]
    for k, i in enumerate(picked, 1):
        t, r = turns[i], rows[i]
        lines += [f"## {k}. `{i}`", "", "**Conversation (last 4):**", ""]
        lines += [f"> {h}" for h in t.history[-4:]] or ["> (none)"]
        lines += ["", f"**User:** {t.query}", "", f"**GOLD:** {' '.join(spans[s] for s in r['gold_spans'])}", "",
                  "**PASSAGES:**", ""]
        lines += [f"{j}. {' '.join(by_pid[p].text.split()[:400])}" for j, p in enumerate(r["top5"], 1)]
        lines += ["", "covered: ", ""]
    (out_dir / "human_check.md").write_text("\n".join(lines))
    key = {i: {n: {"covered": verdicts[n][i], "reason": reasons[n].get(i)} for n in names} for i in picked}
    (out_dir / "human_check.key.json").write_text(json.dumps(key, indent=1, ensure_ascii=False))
    print(f"wrote {out_dir / 'human_check.md'} ({len(picked)} items) and the key")


if __name__ == "__main__":
    main()
