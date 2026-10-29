"""Which withheld turns are still answerable from the passages the pipeline retrieved without the gold document.

    python scripts/judge_withheld.py runs/answers/md2d_validation_n300_s0_withheld.jsonl
    VPQA_JUDGE_MODEL=openai/gpt-5.5 python scripts/judge_withheld.py <run> <out.json>   # second judge

MultiDoc2Dial documents overlap (the same benefit is described on several pages), so withholding the gold
document does not always make a turn unanswerable. For each withheld turn an LLM judge sees the
conversation, the gold grounding text and the top 5 passages retrieved with the gold document withheld,
and says whether the passages contain the information of the gold text needed for the reply. Writes
`<run>.covered.json` (turn id -> bool); scripts/refusal_report.py --covered drops those pairs.
"""
from __future__ import annotations

import json
import argparse
import hashlib
import re
from pathlib import Path

from vpqa import llm
from vpqa.data.multidoc2dial import load_passages, load_turns

RAW = Path("data/raw/multidoc2dial")
DOC = RAW / "multidoc2dial_doc.json"

PROMPT = """A help-desk assistant must reply to the latest user message. The reference reply is grounded in the GOLD text. Do the PASSAGES contain the information from the GOLD text that the reply needs, so that a reply with the same substance could be written from the passages alone? Wording may differ; a passage that only covers a related topic does not count.

Conversation so far:
{history}

Latest user message: {query}

GOLD:
{gold}

PASSAGES:
{passages}

Reply with JSON only: {{"covered": true or false, "reason": "<one short sentence>"}}"""


def parse(reply: str) -> tuple[bool | None, str | None]:
    """The verdict and reason; a reply cut off mid-reason (reasoning used up max_tokens) still has its verdict."""
    m = re.search(r"\{.*\}", reply, re.S)
    try:
        out = json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        out = None
    if isinstance(out, dict):
        value = out.get("covered")
        return (value if type(value) is bool else None), out.get("reason")
    # Only salvage an explicit leading verdict from a truncated JSON object.
    v = re.match(r'\s*\{\s*"covered"\s*:\s*(true|false)\s*[,}]', reply)
    r = re.search(r'"reason"\s*:\s*"(.*)', reply, re.S)
    return (v.group(1) == "true" if v else None), (r.group(1).rstrip() + " [cut off]") if r else None


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    tmp.replace(path)


def judge_prompts(prompts: dict[str, str], out: Path, model: str, *, cache_only: bool = False):
    """Checkpoint each valid verdict; publish coverage only after every item has a verdict.

    Checkpoints are bound to the exact prompts and judge model. A network interruption can be
    resumed, while an invalid reply or absent cache entry remains unknown, never False.
    """
    fingerprint = hashlib.sha256(json.dumps([model, prompts], sort_keys=True).encode()).hexdigest()
    checkpoint = out.with_suffix(".progress.json")
    state = json.loads(checkpoint.read_text()) if checkpoint.exists() else {
        "fingerprint": fingerprint, "model": model, "items": {}}
    if state["fingerprint"] != fingerprint:
        raise ValueError("Judge prompts/model differ from checkpoint; use a different output path")
    missing, invalid = [], []
    for item, prompt in prompts.items():
        if item in state["items"]:
            continue
        try:
            reply = llm.chat([{"role": "user", "content": prompt}], model_name=model, cache_only=cache_only)
        except FileNotFoundError:
            if not cache_only:
                raise
            missing.append(item)
            continue
        verdict, reason = parse(reply)
        if verdict is None:
            invalid.append(item)
            continue
        state["items"][item] = {"covered": verdict, "reason": reason}
        write_json(checkpoint, state)
    state["missing_cache"] = missing
    state["invalid_reply"] = invalid
    write_json(checkpoint, state)
    if len(state["items"]) != len(prompts):
        print(f"Incomplete: {len(state['items'])}/{len(prompts)} valid; "
              f"{len(missing)} cache misses, {len(invalid)} invalid replies. Checkpoint: {checkpoint}")
        return None
    covered = {i: state["items"][i]["covered"] for i in prompts}
    write_json(out.with_suffix(".reasons.json"), {i: state["items"][i]["reason"] for i in prompts})
    write_json(out, covered)
    return covered


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", type=Path)
    ap.add_argument("out", type=Path, nargs="?")
    ap.add_argument("--cache-only", action="store_true", help="recover cached judgments without API calls")
    args = ap.parse_args()
    run = args.run
    rows = [json.loads(l) for l in run.read_text().splitlines()]
    docs = json.loads(DOC.read_text())["doc_data"]
    spans = {f"{doc_id}::{sid}": sp["text_sp"].strip()
             for dom in docs.values() for doc_id, doc in dom.items() for sid, sp in doc["spans"].items()}
    by_pid = {p.pid: p for p in load_passages(DOC)}
    split = "train" if run.name.startswith("md2d_train") else "validation"
    turns = {f"{t.dialog_id}/{t.turn_id}": t for t in load_turns(RAW / f"multidoc2dial_dial_{split}.json", DOC)}
    prompts = {}
    for r in rows:
        t = turns[r["id"]]
        gold = " ".join(spans[s] for s in r["gold_spans"])
        passages = "\n\n".join(f"[{i + 1}] {' '.join(by_pid[p].text.split()[:400])}" for i, p in enumerate(r["top5"]))
        prompts[r["id"]] = PROMPT.format(
            history="\n".join(t.history[-4:]) or "(none)", query=t.query, gold=gold, passages=passages)
    if len(prompts) != len(rows):
        raise ValueError("Duplicate turn IDs in the run")
    out_path = args.out or run.with_suffix(".covered.json")
    covered = judge_prompts(prompts, out_path, llm.judge_model(), cache_only=args.cache_only)
    if covered is None:
        raise SystemExit(2)
    n = len(covered)
    answered = [r for r in rows if r["outcome"] == "answered"]
    print(f"{sum(covered.values())} of {n} withheld turns still covered by the retrieved passages")
    print(f"answered: {len(answered)}, of which covered: {sum(covered[r['id']] for r in answered)}")
    print(f"refused: {n - len(answered)}, of which covered: {sum(covered[r['id']] for r in rows if r['outcome'] != 'answered')}")
    print(f"LLM calls: {llm.TOTALS['calls']}, cached: {llm.TOTALS['cached']}; "
          f"recorded cost (includes cached calls): {llm.TOTALS['cost']:.3f}")


if __name__ == "__main__":
    main()
