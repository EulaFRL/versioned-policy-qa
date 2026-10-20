"""Build the retrieval query for a turn from the current utterance and dialogue history."""
from __future__ import annotations

from vpqa.schema import Turn


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
