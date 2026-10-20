"""Dated versions of federal regulations from the eCFR point-in-time API, as passages with effective dates.

The corpus covers the regulations behind three MultiDoc2Dial domains: Social Security (20 CFR 404, 416),
veterans' benefits (38 CFR 3) and federal student aid (34 CFR 668, 685). For every date on which a part
changed, the whole part is fetched as it read on that date. Each section is cut into passages at its
top-level paragraphs ((a), (b), ...), long paragraphs further at sub-paragraph boundaries. A passage
keeps the same key (section, paragraph, piece) across snapshots; consecutive snapshots with the same text
form one version, valid from the snapshot date it first appears until the date it changes or disappears.
"""
from __future__ import annotations

import gzip
import json
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict
from datetime import date
from pathlib import Path

import httpx

from vpqa.schema import Passage

BASE = "https://www.ecfr.gov/api/versioner/v1"
PARTS = {"ssa": [(20, "404"), (20, "416")], "va": [(38, "3")], "studentaid": [(34, "668"), (34, "685")]}
RAW = Path("data/raw/ecfr")
MAX_WORDS = 250
BASELINE = "2016-12-19"  # earliest date the point-in-time API serves for these parts
SKIP = {"CITA", "PARAUTH", "SECAUTH", "EDNOTE", "SOURCE", "AUTH", "APPRO", "CROSSREF"}
TOP_SKIP = SKIP | {"HEAD", "XREF", "STARS"}  # XREF directly under a section: "Link to an amendment published at ..."
_LABEL = re.compile(r"^\(([a-z]{1,2})\)")
_ROMAN = {"i", "v", "x", "ii", "iv", "vi", "ix", "xi"}


def _get(url: str, params: dict) -> httpx.Response | None:
    """GET with retries on network errors and 5xx; None on 404."""
    for attempt in range(6):
        try:
            r = httpx.get(url, params=params, timeout=180)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r
        except (httpx.TransportError, httpx.HTTPStatusError) as e:
            if attempt == 5 or (isinstance(e, httpx.HTTPStatusError) and e.response.status_code < 500):
                raise
            time.sleep(2 ** attempt)


def list_versions(title: int, part: str) -> list[dict]:
    out, page = [], 1
    while True:
        r = _get(f"{BASE}/versions/title-{title}.json", {"part": part, "page": page})
        out += r.json()["content_versions"]
        if page >= int(r.json()["meta"]["total_pages"]):
            return out
        page += 1


def snapshot_dates(versions: list[dict]) -> list[str]:
    """The baseline plus every later date on which some section of the part changed in substance."""
    changed = {v["date"] for v in versions if v["substantive"] and v["type"] == "section"}
    return sorted({BASELINE} | {d for d in changed if d > BASELINE})


def fetch_part(title: int, part: str, on: str, raw: Path = RAW) -> bytes | None:
    """The part as it read on `on`; None if eCFR has no point-in-time text for that date (before 2016-12)."""
    path = raw / f"{title}-{part}" / f"{on}.xml.gz"
    if path.exists():
        return gzip.decompress(path.read_bytes())
    r = _get(f"{BASE}/full/{on}/title-{title}.xml", {"part": part})
    if r is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(gzip.compress(r.content))
    tmp.replace(path)
    return r.content


def _text(el: ET.Element) -> str:
    """Text of an element without citations, authority notes and editorial notes."""
    if el.tag in SKIP:
        return el.tail or ""
    inner = (el.text or "") + "".join(_text(c) for c in el)
    if el.tag in {"TD", "TH"}:
        inner += " |"
    return inner + (el.tail or "")


def _clean(text: str) -> str:
    return " ".join(text.split())


def _norm(text: str) -> str:
    """Text for comparing versions: eCFR re-typesets dashes and spacing around labels without amending
    anything, e.g. `with -` → `with—`, `amounts. (1)` → `amounts.(1)`."""
    text = re.sub(r"\s*(?:—|--| -)\s*", "—", text)
    return re.sub(r"\s+", "", text)


def _top_label(text: str, current: str | None, prev_label: str | None) -> str | None:
    """The top-level label a paragraph opens, if it opens one: the next letter after `current`. A roman
    numeral right after an arabic sub-paragraph, e.g. (i) after (1), is a sub-paragraph, not (i)."""
    m = _LABEL.match(text)
    if not m:
        return None
    label = m.group(1)
    expected = "a" if current is None else chr(ord(current[-1]) + 1) * len(current)
    if label != expected:
        return None
    if label in _ROMAN and prev_label is not None and prev_label.isdigit():
        return None
    return label


def _children(paras: list[str]) -> list[tuple[str, list[str]]]:
    """Split a long top-level paragraph at its numbered sub-paragraphs (1), (2), ...; if it has none
    (e.g. a list of definitions), at every paragraph, keyed by the paragraph's first words. Keys come from
    the text, not the position, so inserting a sub-paragraph does not rename the ones after it."""
    out: list[tuple[str, list[str]]] = []
    numbered = any(re.match(r"^\(1\)", t) for t in paras[:2])
    n = 0
    for text in paras:
        m = re.match(r"^\((\d+)\)", text)
        if numbered and m and int(m.group(1)) == n + 1:
            n += 1
            out.append((f"({n})", [text]))
        elif not numbered and out and not text.startswith("("):
            out.append((":" + "_".join(re.sub(r"[^\w ]", "", text).lower().split()[:6]), [text]))
        elif out:
            out[-1][1].append(text)
        else:
            out.append(("", [text]))
    return out


def section_passages(sec: ET.Element) -> list[tuple[str, str]]:
    """(paragraph key, text) for one section, heading prepended to every passage."""
    heading = _clean(_text(sec.find("HEAD"))) if sec.find("HEAD") is not None else sec.get("N", "")
    groups: list[tuple[str, list[str]]] = [("intro", [])]
    current, prev = None, None
    for el in sec:
        if el.tag in TOP_SKIP:
            continue
        text = _clean(_text(el))
        if not text:
            continue
        label = _top_label(text, current, prev) if el.tag == "P" else None
        if label:
            current = label
            groups.append((label, []))
        m = re.match(r"^\((\w{1,4})\)", text)
        prev = m.group(1) if m else prev
        groups[-1][1].append(text)
    out: list[tuple[str, str]] = []
    for label, paras in groups:
        if not paras:
            continue
        parts = [(label, paras)] if _words(paras) <= MAX_WORDS else [(label + k, ps) for k, ps in _children(paras)]
        for key, ps in parts:
            # a sub-paragraph still too long is cut by words; only these keys depend on position
            pieces, buf = [], []
            for p in ps:
                if buf and _words(buf) + len(p.split()) > MAX_WORDS:
                    pieces.append(buf)
                    buf = []
                buf.append(p)
            pieces.append(buf)
            for i, piece in enumerate(pieces):
                out.append((key if i == 0 else f"{key}#{i}", f"{heading}\n{' '.join(piece)}"))
    seen: dict[str, int] = {}
    for i, (key, text) in enumerate(out):
        seen[key] = seen.get(key, 0) + 1
        if seen[key] > 1:
            out[i] = (f"{key}~{seen[key]}", text)
    return out


def _words(paras: list[str]) -> int:
    return sum(len(p.split()) for p in paras)


def parse_part(xml: bytes) -> dict[str, str]:
    """{passage key: text} for a part as fetched on one date; key is `section::paragraph`, e.g. `685.209::a(3)`."""
    root = ET.fromstring(xml)
    out = {}
    for sec in root.iter("DIV8"):
        if sec.get("TYPE") != "SECTION":
            continue
        for key, text in section_passages(sec):
            out[f"{sec.get('N')}::{key}"] = text
    return out


def build_versions(snapshots: list[tuple[str, dict[str, str]]], title: int, part: str, domain: str) -> list[Passage]:
    """Merge dated snapshots of one part into versioned passages. A version is valid from its first
    snapshot date until (excluding) the date its key changes text or disappears; open-ended if current.
    Changes in typesetting only (see `_norm`) do not start a new version."""
    open_: dict[str, tuple[str, str]] = {}  # key -> (text, effective_from)
    done: list[Passage] = []

    def close(key: str, until: str | None) -> None:
        text, start = open_.pop(key)
        section = key.split("::")[0]
        done.append(Passage(pid=f"{title}cfr{key}@{start}", doc_id=f"{title} CFR {section}", domain=domain,
                            text=text, effective_from=date.fromisoformat(start),
                            effective_to=date.fromisoformat(until) if until else None))

    for on, units in snapshots:
        for key in [k for k in open_ if k not in units or _norm(units[k]) != _norm(open_[k][0])]:
            close(key, on)
        for key, text in units.items():
            if key not in open_:
                open_[key] = (text, on)
    for key in list(open_):
        close(key, None)
    return done


def load_versions(path: Path = Path("data/processed/ecfr_versions.jsonl")) -> list[Passage]:
    passages = []
    for line in path.read_text().splitlines():
        d = json.loads(line)
        for k in ("effective_from", "effective_to"):
            d[k] = date.fromisoformat(d[k]) if d[k] else None
        passages.append(Passage(**d))
    return passages


def save_versions(passages: list[Passage], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for p in passages:
            d = asdict(p)
            for k in ("effective_from", "effective_to"):
                d[k] = d[k].isoformat() if d[k] else None
            f.write(json.dumps(d) + "\n")
