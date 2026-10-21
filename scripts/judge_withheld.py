"""Which withheld turns are still answerable from the passages the pipeline retrieved without the gold document.

    python scripts/judge_withheld.py runs/answers/md2d_validation_n300_s0_withheld.jsonl

MultiDoc2Dial documents overlap (the same benefit is described on several pages), so withholding the gold
document does not always make a turn unanswerable. For each withheld turn an LLM judge sees the
conversation, the gold grounding text and the top 5 passages retrieved with the gold document withheld,
and says whether the passages contain the information of the gold text needed for the reply. Writes
`<run>.covered.json` (turn id -> bool); scripts/refusal_report.py --covered drops those pairs.
"""
from __future__ import annotations

import json
import sys
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


def main() -> None:
    run = Path(sys.argv[1])
    rows = [json.loads(l) for l in run.read_text().splitlines()]
    docs = json.loads(DOC.read_text())["doc_data"]
    spans = {f"{doc_id}::{sid}": sp["text_sp"].strip()
             for dom in docs.values() for doc_id, doc in dom.items() for sid, sp in doc["spans"].items()}
    by_pid = {p.pid: p for p in load_passages(DOC)}
    turns = {f"{t.dialog_id}/{t.turn_id}": t for t in load_turns(RAW / "multidoc2dial_dial_validation.json", DOC)}
    covered, reasons = {}, {}
    for r in rows:
        t = turns[r["id"]]
        gold = " ".join(spans[s] for s in r["gold_spans"])
        passages = "\n\n".join(f"[{i + 1}] {' '.join(by_pid[p].text.split()[:400])}" for i, p in enumerate(r["top5"]))
        out = llm.chat_json([{"role": "user", "content": PROMPT.format(
            history="\n".join(t.history[-4:]) or "(none)", query=t.query, gold=gold, passages=passages)}])
        covered[r["id"]] = bool(out and out.get("covered"))
        reasons[r["id"]] = (out or {}).get("reason")
    out_path = run.with_suffix(".covered.json")
    out_path.write_text(json.dumps(covered, indent=1))
    n = len(covered)
    answered = [r for r in rows if r["outcome"] == "answered"]
    print(f"{sum(covered.values())} of {n} withheld turns still covered by the retrieved passages")
    print(f"answered: {len(answered)}, of which covered: {sum(covered[r['id']] for r in answered)}")
    print(f"refused: {n - len(answered)}, of which covered: {sum(covered[r['id']] for r in rows if r['outcome'] != 'answered')}")
    print(f"LLM cost: {llm.TOTALS['cost']:.3f}")


if __name__ == "__main__":
    main()
