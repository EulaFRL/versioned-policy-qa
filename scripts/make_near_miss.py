"""Near-miss unanswerable turns: the latest user message is rewritten to ask about a case the documents
do not cover, while staying on the topic of the original turn.

    python scripts/make_near_miss.py --split train --n 200 --seed 1
    python scripts/make_near_miss.py --split validation --n 300 --seed 0

Withholding the gold document (vpqa.eval.refusal) tests a turn whose evidence is missing altogether; a
near-miss turn tests the harder case where retrieval returns the right page, but that page answers a
neighbouring question (another condition, another group of people, a detail it never states). For each
sampled answerable turn:

1. a writer model rewrites the latest user message, given the conversation and the gold passages,
   changing one condition so that the documents no longer give the answer;
2. the pipeline retrieves the top 5 passages for the rewritten turn;
3. a checker model from another family sees the rewritten turn, every passage of the gold document and the
   top 5 passages, and says whether they answer it. Only turns the checker calls unanswered are kept.

Validation uses the turns of scripts/eval_answers.py (same n and seed), so near-miss and original turns
pair 1:1 by id. Writes data/benchmark/md2d_near_miss_<split>.jsonl.
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import replace
from pathlib import Path

import yaml

from vpqa import llm
from vpqa.data.multidoc2dial import load_passages, load_turns
from vpqa.pipeline.dag import Pipeline

RAW = Path("data/raw/multidoc2dial")
DOC = RAW / "multidoc2dial_doc.json"

WRITE = """Below is a conversation between a user and the help desk of a public agency, and the passages from the agency's documents that the help desk used to reply to the latest user message.

Rewrite the latest user message so that:
- it stays on the same topic and still sounds like this user in this conversation;
- it asks about one specific condition, case, group of people, amount, deadline or detail that the text does NOT state, so an honest help desk relying only on the documents would have to say the documents do not cover it;
- the change is small and natural, not absurd or out of scope (no unrelated topics, no other countries, no requests for opinions).

Conversation so far:
{history}

Latest user message: {query}

Passages the reply was grounded in:
{gold}

Reply with JSON only: {{"question": "<the rewritten latest user message>", "changed": "<what you changed and why the text does not answer it, one sentence>"}}"""

CHECK = """A help-desk assistant must reply to the latest user message using only the documents below. Do the documents state the information needed to answer the message itself (not just a related question)? If the documents only cover a neighbouring case, a different group of people, or leave the asked-about detail unstated, they do not answer it. A reply that asks the user a clarifying question does not count as an answer.

Conversation so far:
{history}

Latest user message: {query}

DOCUMENTS:
{docs}

Reply with JSON only: {{"answered": true or false, "reason": "<one short sentence>"}}"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="validation")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--writer", default="anthropic/claude-sonnet-5.5")
    ap.add_argument("--checker", default="openai/gpt-5-mini")
    ap.add_argument("--doc-words", type=int, default=4000)
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--router", type=Path, default=Path("runs/router/router.pkl"))
    ap.add_argument("--out", type=Path, default=Path("data/benchmark"))
    args = ap.parse_args()

    turns = [t for t in load_turns(RAW / f"multidoc2dial_dial_{args.split}.json", DOC) if t.answerable]
    turns = random.Random(args.seed).sample(turns, min(args.n, len(turns)))
    passages = load_passages(DOC)
    by_doc: dict[str, list] = {}
    for p in passages:
        by_doc.setdefault(p.doc_id, []).append(p)
    pipe = Pipeline(passages, yaml.safe_load(open(args.config)), Path("data/index/BAAI__bge-m3.npy"), args.router,
                    timeouts=False)

    rows = []
    for t in turns:
        history = "\n".join(t.history[-4:]) or "(none)"
        w = llm.chat_json([{"role": "user", "content": WRITE.format(
            history=history, query=t.query, gold="\n\n".join(pipe.by_pid[g].text for g in t.gold_pids))}],
            model_name=args.writer, max_tokens=400) or {}
        question = str(w.get("question") or "").strip()
        row = {"id": f"{t.dialog_id}/{t.turn_id}", "split": args.split, "query": t.query, "question": question,
               "changed": w.get("changed"), "kept": False, "reason": None}
        if question:
            state, _ = pipe.run(replace(t, query=question))
            gold_docs = list(dict.fromkeys(pipe.by_pid[g].doc_id for g in t.gold_pids))
            shown = [p for d in gold_docs for p in by_doc[d]]
            shown += [pipe.by_pid[p] for p in state["ranked"][:5] if pipe.by_pid[p].doc_id not in gold_docs]
            text = "\n\n".join(p.text for p in shown)
            text = " ".join(text.split(" ")[:args.doc_words])
            c = llm.chat_json([{"role": "user", "content": CHECK.format(history=history, query=question, docs=text)}],
                              model_name=args.checker, max_tokens=4000) or {}
            row["kept"] = c.get("answered") is False
            row["reason"] = c.get("reason")
        rows.append(row)
        if len(rows) % 25 == 0:
            print(f"{len(rows)} / {len(turns)}, kept {sum(r['kept'] for r in rows)}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"md2d_near_miss_{args.split}.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{sum(r['kept'] for r in rows)} of {len(rows)} rewritten turns kept as unanswerable; "
          f"cost {llm.TOTALS['cost']:.3f} {llm.TOTALS['cost_by_model']}")


if __name__ == "__main__":
    main()
