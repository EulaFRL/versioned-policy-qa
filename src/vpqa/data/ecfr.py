"""Fetch dated versions of eCFR sections to build the version-conflict split."""
from __future__ import annotations

from datetime import date

import httpx

BASE = "https://www.ecfr.gov/api/versioner/v1"


def list_versions(title: int, part: str) -> list[dict]:
    r = httpx.get(f"{BASE}/versions/title-{title}.json", params={"part": part}, timeout=30)
    r.raise_for_status()
    return r.json()["content_versions"]


def fetch_full(title: int, on: date, part: str) -> str:
    r = httpx.get(f"{BASE}/full/{on.isoformat()}/title-{title}.xml", params={"part": part}, timeout=60)
    r.raise_for_status()
    return r.text
