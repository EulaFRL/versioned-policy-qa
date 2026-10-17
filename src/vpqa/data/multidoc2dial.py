"""Load MultiDoc2Dial into the common Turn/Passage schema.

TODO: confirm field names against the released JSON before relying on this.
"""
from __future__ import annotations

import json
from pathlib import Path

from vpqa.schema import Passage, Turn


def load_passages(doc_path: Path) -> list[Passage]:
    raw = json.loads(doc_path.read_text())
    passages = []
    for domain, docs in raw["doc_data"].items():
        for doc_id, doc in docs.items():
            for span_id, span in doc["spans"].items():
                passages.append(Passage(
                    pid=f"{doc_id}::{span_id}",
                    doc_id=doc_id,
                    domain=domain,
                    text=span["text_sp"],
                ))
    return passages


def load_turns(dial_path: Path) -> list[Turn]:
    raise NotImplementedError("map user turns + grounding references to Turn")
