"""Load MultiDoc2Dial into the common Turn/Passage schema.

The retrieval unit is one of
- `section` (default): a document section (`id_sec`), i.e. a heading or one paragraph/list item, with
  its heading path prepended so a short paragraph still carries its topic;
- `heading`: consecutive spans merged while they share a section or a title, the "structure"
  segmentation of the MultiDoc2Dial paper (4,110 passages);
- `token`: the document text cut every 100 space-separated tokens, the paper's "token" segmentation
  (4,283 passages), with the document title prepended.
A Turn is one agent response; its query is the preceding user utterance, its gold spans are the agent's
grounding spans and its gold passages the units holding them. Turns whose agent act is
`respond_no_solution` are marked unanswerable.
"""
from __future__ import annotations

import json
from pathlib import Path

from vpqa.schema import Passage, Turn

UNANSWERABLE_DA = "respond_no_solution"
UNITS = ("section", "heading", "token")
TOKEN_WINDOW = 100


def _clean(text: str) -> str:
    return " ".join(text.split())


def _heading(span: dict) -> str:
    return " > ".join(_clean(t) for t in [*(p["text"] for p in span["parent_titles"]), span["title"]] if _clean(t))


def _with_heading(heading: str, body: str) -> str:
    return body if not heading or body == heading else f"{heading}\n{body}"


def _doc_units(doc: dict, unit: str) -> list[tuple[str, str, list[str]]]:
    """(unit id within the document, text, id_sp of the spans it holds) for one document."""
    if unit == "section":
        sections: dict[str, list[dict]] = {}
        for span in doc["spans"].values():
            sections.setdefault(span["id_sec"], []).append(span)
        return [(sec_id, _with_heading(_heading(spans[0]), _clean(spans[0]["text_sec"])), [s["id_sp"] for s in spans])
                for sec_id, spans in sections.items()]
    spans = sorted(doc["spans"].values(), key=lambda s: int(s["id_sp"]))
    if unit == "heading":
        groups: list[list[dict]] = []
        for span in spans:
            prev = groups[-1][-1] if groups else None
            if prev and (prev["id_sec"] == span["id_sec"] or prev["title"].strip() == span["title"].strip()):
                groups[-1].append(span)
            else:
                groups.append([span])
        return [(f"h{i}", _with_heading(_heading(g[0]), _clean(" ".join(s["text_sp"] for s in g))), [s["id_sp"] for s in g])
                for i, g in enumerate(groups)]
    if unit == "token":
        title = _clean(doc["title"].rsplit("#", 1)[0])
        tokens, starts, pos = doc["doc_text"].split(" "), [], 0
        for tok in tokens:
            starts.append(pos)
            pos += len(tok) + 1
        units = []
        for i in range(0, len(tokens), TOKEN_WINDOW):
            body = _clean(" ".join(tokens[i:i + TOKEN_WINDOW]))
            if not body:
                continue
            lo, hi = starts[i], starts[min(i + TOKEN_WINDOW, len(tokens)) - 1] + len(tokens[min(i + TOKEN_WINDOW, len(tokens)) - 1])
            held = [s["id_sp"] for s in spans if s["start_sp"] < hi and s["end_sp"] > lo]
            units.append((f"t{len(units)}", f"{title}\n{body}", held))
        return units
    raise ValueError(f"unknown unit: {unit}")


def _units(doc_path: Path, unit: str):
    raw = json.loads(Path(doc_path).read_text())
    for domain, docs in raw["doc_data"].items():
        for doc_id, doc in docs.items():
            for uid, text, span_ids in _doc_units(doc, unit):
                yield Passage(pid=f"{doc_id}::{uid}", doc_id=doc_id, domain=domain, text=text), span_ids


def load_passages(doc_path: Path, unit: str = "section") -> list[Passage]:
    return [p for p, _ in _units(doc_path, unit)]


def span_index(doc_path: Path, unit: str = "section") -> dict[tuple[str, str], list[str]]:
    """Map (doc_id, id_sp) to the pids of the units holding that span (two for a span cut by a token window)."""
    index: dict[tuple[str, str], list[str]] = {}
    for p, span_ids in _units(doc_path, unit):
        for sp in span_ids:
            index.setdefault((p.doc_id, sp), []).append(p.pid)
    return index


def load_turns(dial_path: Path, doc_path: Path, unit: str = "section") -> list[Turn]:
    raw = json.loads(Path(dial_path).read_text())
    span_to_pids = span_index(doc_path, unit)
    turns = []
    for domain, dialogs in raw["dial_data"].items():
        for dialog in dialogs:
            utts = dialog["turns"]
            for i, t in enumerate(utts):
                if t["role"] != "agent" or i == 0 or utts[i - 1]["role"] != "user":
                    continue
                answerable = t["da"] != UNANSWERABLE_DA
                refs = [(r["doc_id"], r["id_sp"]) for r in t["references"]] if answerable else []
                turns.append(Turn(
                    dialog_id=dialog["dial_id"],
                    turn_id=t["turn_id"],
                    domain=domain,
                    history=[f"{u['role']}: {_clean(u['utterance'])}" for u in utts[: i - 1]],
                    query=_clean(utts[i - 1]["utterance"]),
                    gold_pids=sorted({pid for ref in refs for pid in span_to_pids[ref]}),
                    gold_spans=sorted({f"{d}::{sp}" for d, sp in refs}),
                    answerable=answerable,
                    user_act=utts[i - 1]["da"],
                ))
    return turns
