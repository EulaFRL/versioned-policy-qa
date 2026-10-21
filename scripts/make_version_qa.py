"""Build the version-conflict question set from amended eCFR passages.

For a pair of consecutive versions of one passage, an LLM first checks that both are the same provision
(paragraphs get relettered, so one label can hold unrelated rules over time), then writes a question whose
answer changed with the amendment. The question is kept only if each version, given alone, answers it and the two answers differ
in substance (two more LLM calls and a judge). A kept question yields two items, one dated inside each
version's validity window; the gold passage is the version in force on that date. Splits are by section.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

from vpqa import llm
from vpqa.data.ecfr import _norm, load_versions
from vpqa.schema import Passage

GEN = """Below are two versions of the same paragraph of the U.S. Code of Federal Regulations.

OLD (in force {old_from} to {old_to}):
{old}

NEW (in force from {new_from}):
{new}

Write one question a member of the public might ask whose correct answer under OLD differs from its correct
answer under NEW because of this amendment.
- Ask about something BOTH versions state explicitly but differently (an amount, a deadline or period, an
  age, a condition, who decides, a required step). Each version alone must answer it. Do not ask about a
  provision that only one version contains.
- Do not mention versions, dates of amendment, "old"/"new", "currently", or section or paragraph numbers.
- Make it self-contained (name the program or benefit) and phrase it as a real person would, in one sentence.
- If no such question exists (the amendment only adds or removes a provision, or is a renumbering,
  cross-reference or editorial change), return {{"skip": true}}.
Reply with the JSON object only, no other text: {{"question": "...", "answer_old": "...", "answer_new": "..."}},
with short answers."""

ANSWER = """Answer the question using only the regulation text below. Answer in one or two sentences. If the text does
not answer the question, answer exactly NOT_ANSWERED.

Text:
{text}

Question: {question}

Reply with the JSON object only, no other text: {{"answer": "..."}}"""

SAME = """Below are two consecutive texts found under the same paragraph label of a U.S. federal regulation.

OLD:
{old}

NEW:
{new}

Is NEW an amended version of the same provision as OLD (same subject, with some rules changed), or a different
provision that now sits under the same label (for example after paragraphs were renumbered, or the paragraph was
replaced by a rule on another subject)?
Reply with the JSON object only, no other text: {{"same_provision": true}} or {{"same_provision": false}}"""

JUDGE = """Question: {question}
Answer A: {a}
Answer B: {b}

Do A and B give materially different answers (a different amount, date, period, condition, eligibility rule,
requirement or procedure)? Differences in wording alone do not count.
Reply with the JSON object only, no other text: {{"different": true}} or {{"different": false}}"""


def changed_words(old: str, new: str) -> int:
    a, b = old.split(), new.split()
    ops = difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes()
    return sum(max(i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in ops
               if op != "equal" and _norm(" ".join(a[i1:i2])) != _norm(" ".join(b[j1:j2])))


def candidate_pairs(versions: list[Passage], min_changed: int, min_words: int) -> list[tuple[Passage, Passage]]:
    by_key: dict[str, list[Passage]] = defaultdict(list)
    for p in versions:
        by_key[p.pid.split("@")[0]].append(p)
    pairs = []
    for vs in by_key.values():
        vs.sort(key=lambda p: p.effective_from)
        for old, new in zip(vs, vs[1:]):
            if old.effective_to != new.effective_from:
                continue
            if min(len(old.text.split()), len(new.text.split())) < min_words or "[Reserved]" in old.text + new.text:
                continue
            if changed_words(old.text, new.text) >= min_changed:
                pairs.append((old, new))
    return pairs


def split_of(section: str, dev_share: float) -> str:
    return "dev" if int(hashlib.md5(section.encode()).hexdigest(), 16) % 1000 < dev_share * 1000 else "test"


def body(p: Passage) -> str:
    return p.text.split("\n", 1)[-1]


def make_items(old: Passage, new: Passage) -> dict:
    same = llm.chat_json([{"role": "user", "content": SAME.format(old=body(old), new=body(new))}]) or {}
    if same.get("same_provision") is not True:
        return {"status": "different_provision"}
    q = llm.chat_json([{"role": "user", "content": GEN.format(
        old_from=old.effective_from, old_to=old.effective_to, old=old.text, new_from=new.effective_from, new=new.text)}],
        max_tokens=1500)
    if not q or q.get("skip") or not q.get("question"):
        return {"status": "skipped_by_generator"} if q else {"status": "bad_json"}
    question = q["question"].strip()
    a_old = (llm.chat_json([{"role": "user", "content": ANSWER.format(text=old.text, question=question)}]) or {}).get("answer", "")
    a_new = (llm.chat_json([{"role": "user", "content": ANSWER.format(text=new.text, question=question)}]) or {}).get("answer", "")
    rec = {"question": question, "gen": q, "answer_old": a_old, "answer_new": a_new}
    if "NOT_ANSWERED" in a_old or "NOT_ANSWERED" in a_new or not a_old or not a_new:
        return {**rec, "status": "not_answered"}
    judged = llm.chat_json([{"role": "user", "content": JUDGE.format(question=question, a=a_old, b=a_new)}]) or {}
    if judged.get("different") is not True:
        return {**rec, "status": "same_answer"}
    return {**rec, "status": "kept"}


def as_of_in(p: Passage, rng: random.Random, today: date) -> date:
    end = p.effective_to or today
    return p.effective_from + timedelta(days=rng.randrange(max((end - p.effective_from).days, 1)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--versions", type=Path, default=Path("data/processed/ecfr_versions.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("data/benchmark/ecfr_version_qa.jsonl"))
    ap.add_argument("--per-domain", type=int, help="cap on pairs sent to the generator per domain")
    ap.add_argument("--per-section", type=int, help="cap on pairs per section")
    ap.add_argument("--min-changed", type=int, default=3)
    ap.add_argument("--min-words", type=int, default=25)
    ap.add_argument("--dev-share", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--today", type=date.fromisoformat, default=date(2026, 10, 1),
                    help="upper end of the validity window of versions still in force")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="only report candidate pairs")
    args = ap.parse_args()

    versions = load_versions(args.versions)
    pairs = candidate_pairs(versions, args.min_changed, args.min_words)
    rng = random.Random(args.seed)
    rng.shuffle(pairs)
    chosen, per_section, keys = defaultdict(list), Counter(), set()
    for old, new in pairs:  # one pair per paragraph, so no two questions come from the same rule text
        key = old.pid.split("@")[0]
        if (key in keys or (args.per_section and per_section[old.doc_id] >= args.per_section)
                or (args.per_domain and len(chosen[old.domain]) >= args.per_domain)):
            continue
        keys.add(key)
        per_section[old.doc_id] += 1
        chosen[old.domain].append((old, new))
    print("candidate pairs:", Counter(o.domain for o, _ in pairs), "sections:", len({o.doc_id for o, _ in pairs}))
    print("sent to generator:", {d: len(v) for d, v in chosen.items()})
    if args.dry_run:
        return

    jobs = [pair for d in sorted(chosen) for pair in chosen[d]]
    with ThreadPoolExecutor(args.workers) as ex:
        results = list(ex.map(lambda pr: make_items(*pr), jobs))
    print("outcomes:", Counter(r["status"] for r in results))

    stale = defaultdict(list)
    for p in versions:
        stale[p.pid.split("@")[0]].append(p.pid)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with args.out.open("w") as f, args.out.with_suffix(".log.jsonl").open("w") as log:
        for (old, new), r in zip(jobs, results):
            log.write(json.dumps({"old": old.pid, "new": new.pid, **r}) + "\n")
            if r["status"] != "kept":
                continue
            key = old.pid.split("@")[0]
            for which, gold, answer in (("old", old, r["answer_old"]), ("new", new, r["answer_new"])):
                f.write(json.dumps({
                    "qid": f"{key}@{gold.effective_from}", "question": r["question"], "domain": old.domain,
                    "section": old.doc_id, "split": split_of(old.doc_id, args.dev_share), "version": which,
                    "as_of": as_of_in(gold, rng, args.today).isoformat(), "gold_pid": gold.pid,
                    "other_versions": [p for p in stale[key] if p != gold.pid], "answer": answer,
                }) + "\n")
                n += 1
    print(f"wrote {n} items to {args.out}; model {llm.model()}")


if __name__ == "__main__":
    main()
