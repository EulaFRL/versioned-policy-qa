"""Load MultiDoc2Dial into the common Turn/Passage schema.

Retrieval unit is a document section (`id_sec`): a heading or a paragraph/list item, with its heading
path prepended so a short paragraph still carries its topic. A Turn is one agent response; its query is
the preceding user utterance and its gold passages are the sections holding the agent's grounding spans.
Turns whose agent act is `respond_no_solution` are marked unanswerable.
"""
from __future__ import annotations

import json
from pathlib import Path

from vpqa.schema import Passage, Turn

UNANSWERABLE_DA = "respond_no_solution"


def _clean(text: str) -> str:
    return " ".join(text.split())


def load_passages(doc_path: Path) -> list[Passage]:
    raw = json.loads(Path(doc_path).read_text())
    passages = []
    for domain, docs in raw["doc_data"].items():
        for doc_id, doc in docs.items():
            sections: dict[str, dict] = {}
            for span in doc["spans"].values():
                sections.setdefault(span["id_sec"], span)
            for sec_id, span in sections.items():
                body = _clean(span["text_sec"])
                heading = " > ".join(_clean(t) for t in [*(p["text"] for p in span["parent_titles"]), span["title"]] if _clean(t))
                text = body if not heading or body == heading else f"{heading}\n{body}"
                passages.append(Passage(pid=f"{doc_id}::{sec_id}", doc_id=doc_id, domain=domain, text=text))
    return passages


def section_index(doc_path: Path) -> dict[tuple[str, str], str]:
    """Map (doc_id, id_sp) to the pid of the section containing that span."""
    raw = json.loads(Path(doc_path).read_text())
    return {
        (doc_id, span["id_sp"]): f"{doc_id}::{span['id_sec']}"
        for docs in raw["doc_data"].values()
        for doc_id, doc in docs.items()
        for span in doc["spans"].values()
    }


def load_turns(dial_path: Path, doc_path: Path) -> list[Turn]:
    raw = json.loads(Path(dial_path).read_text())
    span_to_pid = section_index(doc_path)
    turns = []
    for domain, dialogs in raw["dial_data"].items():
        for dialog in dialogs:
            utts = dialog["turns"]
            for i, t in enumerate(utts):
                if t["role"] != "agent" or i == 0 or utts[i - 1]["role"] != "user":
                    continue
                answerable = t["da"] != UNANSWERABLE_DA
                gold = sorted({span_to_pid[(r["doc_id"], r["id_sp"])] for r in t["references"]}) if answerable else []
                turns.append(Turn(
                    dialog_id=dialog["dial_id"],
                    turn_id=t["turn_id"],
                    domain=domain,
                    history=[f"{u['role']}: {_clean(u['utterance'])}" for u in utts[: i - 1]],
                    query=_clean(utts[i - 1]["utterance"]),
                    gold_pids=gold,
                    answerable=answerable,
                    user_act=utts[i - 1]["da"],
                ))
    return turns
