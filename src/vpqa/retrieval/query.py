"""Build the retrieval query for a turn from the current utterance and dialogue history."""
from __future__ import annotations

from typing import Callable

from vpqa.schema import Turn

REWRITE = """Below is a conversation between a user and the help desk of a public agency. Rewrite the user's latest message as one standalone search query for the agency's documents:
- replace references to earlier parts of the conversation (it, that, this benefit, the form) with what they refer to;
- if the message replies to a question from the help desk (yes, no, a detail about the user's situation), say what the user wants to know given that reply;
- keep the user's own words where you can, and add nothing the conversation does not say.

Conversation so far:
{history}

Latest user message: {query}

Reply with the query only, on one line."""


def needs_context(utterance: str, short_words: int) -> bool:
    """Short follow-ups are usually replies to the agent ("yes", "no, it was cancelled") that cannot be
    retrieved on their own; longer ones are usually self-contained questions that history only dilutes.
    The threshold is chosen on the train split (scripts/select_gate.py)."""
    return len(utterance.split()) <= short_words


def build_query(turn: Turn, mode: str = "last", history_turns: int = 4, max_words: int = 96,
                short_words: int = 4) -> str:
    """Current utterance first, then earlier utterances most recent first, truncated to `max_words`
    so the oldest context is dropped first and the current utterance is always kept.

    `last`: current utterance only. `recent`: the last `history_turns` utterances (4 = two user/agent
    rounds). `history`: the whole history (the MultiDoc2Dial retriever convention). `gated`: `recent`
    only when `needs_context`, else `last`. `oracle`: like `gated` but decided by the gold user dialogue
    act; an upper bound for the gate, not a system.
    """
    if mode == "gated":
        mode = "recent" if needs_context(turn.query, short_words) else "last"
    elif mode == "oracle":
        mode = "last" if (turn.user_act or "").startswith("query") else "recent"
    if mode == "last":
        return turn.query
    if mode == "recent":
        context = turn.history[-history_turns:] if history_turns else []
    elif mode == "history":
        context = turn.history
    else:
        raise ValueError(f"unknown query mode: {mode}")
    words = " ".join([turn.query, *reversed(context)]).split()
    return " ".join(words[:max(max_words, len(turn.query.split()))])


def rewrite_query(turn: Turn, chat: Callable[[list[dict]], str], history_turns: int = 4) -> str | None:
    """The latest message as a standalone query written by an LLM from the last `history_turns` utterances;
    None when the reply is empty."""
    history = "\n".join(turn.history[-history_turns:]) if history_turns else ""
    reply = chat([{"role": "user", "content": REWRITE.format(history=history or "(none)", query=turn.query)}])
    lines = [x.strip().strip('"') for x in reply.strip().splitlines() if x.strip()]
    return lines[0] if lines else None
