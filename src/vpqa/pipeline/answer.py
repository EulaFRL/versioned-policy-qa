"""Cited answer generation: the LLM sees the top passages numbered [1]..[n] and must end every statement
with the numbers of the passages that support it.

Passages are numbered rather than shown by pid because pids are long and get miscopied. The reply is
parsed into claims, each the text up to a group of citations (splitting on citations rather than on
periods, which abbreviations such as "U.S." break). Text after the last citation that is not a question,
and citations to numbers that were not shown, are reported as issues rather than silently dropped.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from vpqa.schema import Passage, Turn

NOT_FOUND = "NOT_FOUND"

PROMPT = """You are the help desk of a public agency. Answer the user's latest message using only the numbered passages below.

Rules:
- End every statement with the numbers of the passages that support it in square brackets, e.g. "You can apply online [2]." or "... [1][3]."
- Use only what the passages say. Do not add facts from your own knowledge.
- If the passages do not answer the message, reply with exactly {not_found} and nothing else.
- If the answer depends on the user's situation and the passages cover several cases, you may instead ask one short clarifying question; a question needs no citation.
- At most three sentences.

Passages:
{passages}

Conversation so far:
{history}

Latest user message: {query}"""

FEEDBACK = """A check of your answer found these problems:
{problems}

Rewrite the answer under the same rules. Correct or drop every statement the cited passages do not support, and end every statement with its citations. If what remains does not answer the message, reply with exactly {not_found}."""

_CLAIM = re.compile(r"(.*?)((?:\s*\[\d+\])+)[.!?]?", re.S)
_CITE = re.compile(r"\[(\d+)\]")


@dataclass
class Parsed:
    answer: str
    cited_pids: list[str]
    refused: bool
    issues: list[str] = field(default_factory=list)
    claims: list[tuple[str, list[str]]] = field(default_factory=list)  # (claim text, pids it cites)


def passage_label(p: Passage) -> str:
    if p.effective_from is None:
        return ""
    until = f" until {p.effective_to}" if p.effective_to else ""
    return f"({p.doc_id}, in force from {p.effective_from}{until}) "


def build_prompt(turn: Turn, passages: list[Passage], history_turns: int = 4, passage_words: int = 400) -> str:
    shown = "\n\n".join(f"[{i}] {passage_label(p)}{' '.join(p.text.split()[:passage_words])}"
                        for i, p in enumerate(passages, 1))
    history = "\n".join(turn.history[-history_turns:]) if turn.history and history_turns else "(none)"
    return PROMPT.format(not_found=NOT_FOUND, passages=shown, history=history, query=turn.query)


def parse(reply: str, pids: list[str]) -> Parsed:
    """Map [n] citations back to pids; list uncited statements and citations to numbers not shown."""
    text = reply.strip()
    if not text or text.strip(" .\"'") == NOT_FOUND:
        return Parsed(text, [], refused=True, issues=[] if text else ["empty reply"])
    cited, issues, claims, end = [], [], [], 0
    for m in _CLAIM.finditer(text):
        pids_here = []
        for n in map(int, _CITE.findall(m.group(2))):
            if 1 <= n <= len(pids):
                pids_here.append(pids[n - 1])
            else:
                issues.append(f"unknown citation [{n}]")
        claims.append((m.group(1).strip(), list(dict.fromkeys(pids_here))))
        end = m.end()
    tail = text[end:].strip()
    if tail and not tail.endswith("?"):
        issues.append(f"uncited: {tail[:120]}")
    cited = list(dict.fromkeys(p for _, ps in claims for p in ps))
    return Parsed(text, cited, refused=False, issues=issues, claims=claims)


def first_messages(turn: Turn, passages: list[Passage], **kw) -> list[dict]:
    return [{"role": "user", "content": build_prompt(turn, passages, **kw)}]


def revision_messages(messages: list[dict], previous: str, problems: list[str]) -> list[dict]:
    """The conversation so far plus the rejected answer and what the verifier found wrong with it."""
    feedback = FEEDBACK.format(problems="\n".join(f"- {p}" for p in problems), not_found=NOT_FOUND)
    return [*messages, {"role": "assistant", "content": previous}, {"role": "user", "content": feedback}]
