"""ReAct control group: the same LLM and the same retrieval, but the model decides whether to search, with
what query, in which department, and how often (at most `max_searches` times), then answers.

The search tool runs the retrieval stages of the DAG (hybrid retrieval and rerank) on the model's query;
the department and, on a versioned corpus, the as-of date are tool arguments the model may leave out, in
which case all departments or all versions are searched. Passages withheld for the turn stay withheld.
Passages are numbered across searches in the order they first appear, so a citation [n] means the same
passage throughout, and the final reply is parsed exactly like a DAG answer (vpqa.pipeline.answer).
There is no verifier and no revision: this is the free-planning baseline, not a variant of the DAG.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, replace
from datetime import date
from typing import Callable

from vpqa.pipeline import answer as cited
from vpqa.schema import Turn

SYSTEM = """You are the help desk of a public agency. You can search the agency's documents with the search tool; each passage it returns has a number [n] that stays the same across searches. Decide yourself whether to search, with which query, and how often (at most {max_searches} searches), then answer the user's latest message.

Rules for the answer:
- End every statement with the numbers of the passages that support it in square brackets, e.g. "You can apply online [2]." or "... [1][3]."
- Use only what the passages say. Do not add facts from your own knowledge.
- If the passages do not answer the message, reply with exactly {not_found} and nothing else.
- If the answer depends on the user's situation and the passages cover several cases, you may instead ask one short clarifying question; a question needs no citation.
- At most three sentences."""

USER = """Conversation so far:
{history}

Latest user message: {query}"""


@dataclass
class ReactResult:
    answer: str
    cited_pids: list[str]
    refused: bool
    issues: list[str]
    claims: list[tuple[str, list[str]]] = field(default_factory=list)
    searches: list[dict] = field(default_factory=list)  # query, department, as_of, pids returned
    llm_calls: int = 0
    forced: bool = False  # search budget used up, answer requested without tools


class ReactAgent:
    """`pipe` is a vpqa.pipeline.dag.Pipeline (by_pid, versioned, corpus_filter, retrieve, rerank)."""

    def __init__(self, pipe, chat_tools: Callable[..., dict], max_searches: int = 4,
                 history_turns: int = 4, passage_words: int = 400, top_n: int = 5):
        self.pipe, self.chat_tools = pipe, chat_tools
        self.max_searches, self.history_turns = max_searches, history_turns
        self.passage_words, self.top_n = passage_words, top_n
        self.departments = sorted({p.domain for p in pipe.by_pid.values()})

    def tools(self) -> list[dict]:
        props = {"query": {"type": "string", "description": "What to look for."},
                 "department": {"type": "string", "enum": self.departments,
                                "description": "Search only this department's documents; leave out to search all."}}
        if self.pipe.versioned:
            props["as_of"] = {"type": "string", "description": "YYYY-MM-DD: search only the versions in force on "
                                                               "this date; leave out to search every version."}
        return [{"type": "function", "function": {
            "name": "search", "description": f"Search the agency documents; returns the {self.top_n} most relevant "
                                             "passages, numbered.",
            "parameters": {"type": "object", "properties": props, "required": ["query"]}}}]

    def search(self, turn: Turn, args: dict) -> tuple[list[str], dict]:
        dept = args.get("department") if args.get("department") in self.departments else None
        try:
            as_of = date.fromisoformat(args["as_of"]) if args.get("as_of") else None
        except ValueError:
            as_of = None
        query = str(args.get("query") or turn.query)
        valid = self.pipe.corpus_filter(replace(turn, as_of=as_of))["valid"]
        candidates = self.pipe.retrieve(query, dept, valid)["candidates"]
        ranked = self.pipe.rerank(query, candidates)["ranked"]
        return ranked[:self.top_n], {"query": query, "department": dept, "as_of": args.get("as_of")}

    def run(self, turn: Turn) -> ReactResult:
        history = "\n".join(turn.history[-self.history_turns:]) if turn.history and self.history_turns else "(none)"
        messages = [{"role": "system", "content": SYSTEM.format(max_searches=self.max_searches,
                                                                not_found=cited.NOT_FOUND)},
                    {"role": "user", "content": USER.format(history=history, query=turn.query)}]
        numbered: list[str] = []  # pid of passage [i + 1]
        searches, calls, forced = [], 0, False
        while True:
            budget_left = len(searches) < self.max_searches
            forced = not budget_left
            msg = self.chat_tools(messages, self.tools(), tool_choice="auto" if budget_left else "none")
            calls += 1
            tool_calls = msg.get("tool_calls") or []
            if not tool_calls or not budget_left:
                break
            messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": tool_calls})
            for call in tool_calls:
                try:
                    args = json.loads(call["function"].get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                if len(searches) >= self.max_searches:
                    content = "Search budget used up; answer now."
                else:
                    pids, record = self.search(turn, args)
                    searches.append({**record, "pids": pids})
                    lines = []
                    for pid in pids:
                        if pid not in numbered:
                            numbered.append(pid)
                        p = self.pipe.by_pid[pid]
                        lines.append(f"[{numbered.index(pid) + 1}] {cited.passage_label(p)}"
                                     f"{' '.join(p.text.split()[:self.passage_words])}")
                    content = "\n\n".join(lines) or "No passages found."
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": content})
        parsed = cited.parse(msg.get("content") or "", numbered)
        return ReactResult(parsed.answer, parsed.cited_pids, parsed.refused, parsed.issues, parsed.claims,
                           searches, calls, forced)


def timed(agent: ReactAgent, turn: Turn) -> tuple[ReactResult, float]:
    start = time.perf_counter()
    result = agent.run(turn)
    return result, round((time.perf_counter() - start) * 1000, 1)
