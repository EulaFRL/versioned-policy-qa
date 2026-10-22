"""Check a cited answer before it is shown: format, version, then support.

1. Format: the parse issues of vpqa.pipeline.answer (text after the last citation, citations to passages
   that were not shown) and claims that cite nothing fail the answer.
2. Version: every cited passage must be in force on the turn's as-of date. Retrieval already filters by
   date, so this guards against a stale passage reaching the answer by any other route.
3. Support: one LLM call judges every claim against the passages it cites.
4. Question (with the user's message given as `question`): when every claim is supported, a second call
   judges whether the answer replies to what was asked. An answer can be true to its passages and still be
   about a neighbouring case (another group of people, another condition, a detail the passages never
   state), which the support check alone lets through. It is a call of its own so that the support
   verdicts do not change with it.

The result is a pass/fail and a list of problems phrased so they can be handed back to the answer model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Callable

from vpqa.schema import Passage

JUDGE = """Below are the claims of an answer, each followed by the passages it cites. For each claim, decide whether the cited passages support it: a paraphrase or a summary is fine, but every fact in the claim must be stated in the cited passages, and the claim must not contradict them.

{claims}

Passages:
{passages}

Reply with the JSON object only, no other text: {{"verdicts": [{{"claim": 1, "supported": true, "problem": ""}}, ...]}}, one entry per claim, with "problem" a short reason when "supported" is false."""


ADDRESSES = """A help-desk assistant replied to the latest user message below using the passages it cites. Every fact in the answer is supported by those passages; the question here is only whether the answer, taken as a whole, replies to what the latest user message asks.

It does not when it answers a related but different question: another group of people, another condition or case, or a general statement where the user asked for a specific detail the passages do not give. A clarifying question back to the user counts as replying, and so does a partial answer that gives what the passages state about the thing asked.

{question}

Answer: {answer}

Passages:
{passages}

Reply with the JSON object only, no other text: {{"addresses_question": true, "gap": ""}}, with "gap" what the user asked that the answer does not give when "addresses_question" is false."""


@dataclass
class Verdict:
    passed: bool
    problems: list[str] = field(default_factory=list)


def in_force(p: Passage, on: date | None) -> bool:
    if on is None or p.effective_from is None:
        return True
    return p.effective_from <= on and (p.effective_to is None or on < p.effective_to)


def verify(claims: list[tuple[str, list[str]]], issues: list[str], passages: dict[str, Passage],
           as_of: date | None, judge: Callable[[list[dict]], dict | None], passage_words: int = 400,
           question: str | None = None) -> Verdict:
    problems = list(issues)
    problems += [f"claim {i} cites no passage: {text[:120]}" for i, (text, pids) in enumerate(claims, 1) if not pids]
    for pid in dict.fromkeys(p for _, pids in claims for p in pids):
        if not in_force(passages[pid], as_of):
            p = passages[pid]
            problems.append(f"cites {p.doc_id} in the version from {p.effective_from}, not in force on {as_of}")
    if problems:
        return Verdict(False, problems)
    if not claims:  # a clarifying question only
        return Verdict(True)

    cited = list(dict.fromkeys(p for _, pids in claims for p in pids))
    label = {pid: f"P{i}" for i, pid in enumerate(cited, 1)}
    claim_text = "\n".join(f'{i}. "{text}" cites {", ".join(label[p] for p in pids)}'
                           for i, (text, pids) in enumerate(claims, 1))
    passage_text = "\n\n".join(f"{label[pid]}: {' '.join(passages[pid].text.split()[:passage_words])}" for pid in cited)
    reply = judge([{"role": "user", "content": JUDGE.format(claims=claim_text, passages=passage_text)}]) or {}
    verdicts = {v.get("claim"): v for v in reply.get("verdicts", []) if isinstance(v, dict)}
    for i, (text, _) in enumerate(claims, 1):
        v = verdicts.get(i)
        if v is None:
            problems.append(f"claim {i} was not judged")
        elif v.get("supported") is not True:
            problems.append(f"claim {i} is not supported by its passages ({v.get('problem') or 'no reason'}): {text[:120]}")
    if question is not None and not problems:
        answer = " ".join(text for text, _ in claims)
        reply = judge([{"role": "user", "content": ADDRESSES.format(question=question, answer=answer,
                                                                    passages=passage_text)}]) or {}
        if reply.get("addresses_question") is not True:
            problems.append(f"the answer does not reply to what the user asked ({reply.get('gap') or 'no reason'}); "
                            "if the passages do not give it, the reply is NOT_FOUND")
    return Verdict(not problems, problems)
