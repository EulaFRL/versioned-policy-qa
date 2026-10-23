"""The generation half of the DAG: answer -> verify -> (revise -> verify) x2 -> finalize.

The answer node refuses without an LLM call when nothing was retrieved or the best rerank score is below
min_score (configs `answer.min_score`, picked on train by scripts/select_refusal_gate.py).

The revision loop is unrolled into fixed nodes that are skipped once an answer passes. An answer that still
fails after the last revision, or that could not be generated at all, is replaced by the passage text
itself with a note that the documents did not support a checked answer; a refusal (NOT_FOUND) is kept.
"""
from __future__ import annotations

from typing import Any, Callable

from vpqa.pipeline import answer as cited
from vpqa.pipeline import verifier
from vpqa.pipeline.engine import Node
from vpqa.schema import Passage, Turn

ANSWER_OUT = {"answer": str, "cited_pids": list[str], "refused": bool, "issues": list[str], "claims": list[dict],
              "messages": list[dict]}
VERIFY_OUT = {"verified": bool, "problems": list[str]}
DEGRADED = "I could not check an answer against the documents, so here is the most relevant text as written:"


class Generator:
    def __init__(self, passages: dict[str, Passage], llm: Callable[[list[dict]], str],
                 judge: Callable[[list[dict]], dict | None], top_n: int = 5, max_revisions: int = 2,
                 history_turns: int = 4, passage_words: int = 400, min_score: float | None = None,
                 check_question: bool = False, rules: list[str] | None = None):
        self.by_pid, self.llm, self.judge = passages, llm, judge
        self.top_n, self.max_revisions = top_n, max_revisions
        self.history_turns, self.passage_words = history_turns, passage_words
        self.min_score = min_score  # refuse without asking the LLM when the best rerank score is below this
        self.check_question = check_question  # verifier also judges whether the answer replies to the message
        self.rules = rules or []  # extra lines for the answer prompt's rule list

    def nodes(self, limit: Callable[[str], float | None] = lambda name: None) -> list[Node]:
        nodes = [Node("answer", self.respond, {"turn": Turn, "ranked": list[str], "top_score": float | None}, ANSWER_OUT,
                      limit("answer"), fallback=self.no_answer),
                 Node("verify", self.verify, {"turn": Turn, "refused": bool, "claims": list[dict],
                                              "issues": list[str], "messages": list[dict]}, VERIFY_OUT,
                      limit("verify"), fallback=self.not_verified)]
        for i in range(1, self.max_revisions + 1):
            nodes += [Node(f"revise_{i}", self.revise, {"ranked": list[str], "messages": list[dict], "answer": str,
                                                         "problems": list[str]}, ANSWER_OUT,
                           limit("answer"), fallback=self.keep, skip_if="verified"),
                      Node(f"verify_{i}", self.verify, nodes[1].inputs, VERIFY_OUT,
                           limit("verify"), fallback=self.not_verified, skip_if="verified")]
        nodes.append(Node("finalize", self.finalize, {"ranked": list[str], "answer": str, "cited_pids": list[str],
                                                      "refused": bool, "verified": bool},
                          {"final": str, "final_pids": list[str], "outcome": str}))
        return nodes

    def shown(self, ranked: list[str]) -> list[str]:
        return ranked[:self.top_n]

    def ask(self, messages: list[dict], ranked: list[str]) -> dict[str, Any]:
        parsed = cited.parse(self.llm(messages), self.shown(ranked))
        return {"answer": parsed.answer, "cited_pids": parsed.cited_pids, "refused": parsed.refused,
                "issues": parsed.issues, "claims": [{"text": t, "pids": ps} for t, ps in parsed.claims],
                "messages": messages}

    def respond(self, turn: Turn, ranked: list[str], top_score: float | None) -> dict:
        if not ranked or (self.min_score is not None and top_score is not None and top_score < self.min_score):
            return {**self.no_answer(turn, ranked, top_score), "answer": cited.NOT_FOUND, "refused": True,
                    "issues": []}
        passages = [self.by_pid[p] for p in self.shown(ranked)]
        return self.ask(cited.first_messages(turn, passages, history_turns=self.history_turns,
                                              passage_words=self.passage_words, rules=self.rules), ranked)

    def no_answer(self, turn: Turn, ranked: list[str], top_score: float | None = None) -> dict:
        """Fallback: no answer to check or revise; finalize shows the passage text instead."""
        return {"answer": "", "cited_pids": [], "refused": False, "issues": ["no answer was generated"],
                "claims": [], "messages": []}

    def verify(self, turn: Turn, refused: bool, claims: list[dict], issues: list[str], messages: list[dict]) -> dict:
        if refused:
            return {"verified": True, "problems": []}
        if not messages:  # nothing was generated, so nothing can be revised either
            return {"verified": False, "problems": issues}
        question = None
        if self.check_question:
            history = "\n".join(turn.history[-self.history_turns:]) if turn.history and self.history_turns else "(none)"
            question = f"Conversation so far:\n{history}\n\nLatest user message: {turn.query}"
        v = verifier.verify([(c["text"], c["pids"]) for c in claims], issues, self.by_pid, turn.as_of,
                            self.judge, self.passage_words, question)
        return {"verified": v.passed, "problems": v.problems}

    def not_verified(self, **_) -> dict:
        return {"verified": False, "problems": ["the verifier did not finish"]}

    def revise(self, ranked: list[str], messages: list[dict], answer: str, problems: list[str]) -> dict:
        if not messages:
            return self.keep(ranked, messages, answer, problems)
        return self.ask(cited.revision_messages(messages, answer, problems), ranked)

    def keep(self, ranked: list[str], messages: list[dict], answer: str, problems: list[str]) -> dict:
        """Fallback for a revision that did not finish: an unrevisable, unverified answer."""
        return {"answer": answer, "cited_pids": [], "refused": False, "issues": problems, "claims": [],
                "messages": []}

    def finalize(self, ranked: list[str], answer: str, cited_pids: list[str], refused: bool, verified: bool) -> dict:
        if refused:
            return {"final": cited.NOT_FOUND, "final_pids": [], "outcome": "refused"}
        if verified:
            return {"final": answer, "final_pids": cited_pids, "outcome": "answered"}
        pid = (cited_pids or ranked or [None])[0]
        if pid is None:
            return {"final": cited.NOT_FOUND, "final_pids": [], "outcome": "refused"}
        text = " ".join(self.by_pid[pid].text.split()[:120])
        return {"final": f'{DEGRADED} "{text}" [1]', "final_pids": [pid], "outcome": "degraded"}
