from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


@dataclass
class Passage:
    pid: str
    doc_id: str
    domain: str
    text: str
    effective_from: date | None = None
    effective_to: date | None = None


@dataclass
class Turn:
    dialog_id: str
    turn_id: int
    domain: str
    history: list[str]
    query: str
    gold_pids: list[str] = field(default_factory=list)
    gold_spans: list[str] = field(default_factory=list)  # "doc_id::id_sp" grounding spans, independent of the retrieval unit
    answerable: bool = True
    as_of: date | None = None
    user_act: str | None = None
