"""Chat completions through any OpenAI-compatible API, configured by VPQA_API_BASE / VPQA_API_KEY / VPQA_MODEL.

Responses are cached on disk by (model, messages, temperature), so reruns cost nothing and are reproducible.
Each cached response keeps the token usage (and, on OpenRouter, the cost) of the call that produced it;
TOTALS sums them over every call in this process, cached or not, so a rerun reports what the run cost.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path

import httpx

CACHE = Path("data/cache/llm")
TOTALS = {"calls": 0, "cached": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0}


def _count(usage: dict | None, cached: bool) -> None:
    TOTALS["calls"] += 1
    TOTALS["cached"] += cached
    for k in ("prompt_tokens", "completion_tokens", "cost"):
        TOTALS[k] += (usage or {}).get(k) or 0


def model() -> str:
    return os.environ["VPQA_MODEL"]


def chat(messages: list[dict], temperature: float = 0.0, max_tokens: int = 800, cache: Path = CACHE) -> str:
    body = {"model": model(), "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    key = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    path = cache / key[:2] / f"{key}.json"
    if path.exists():
        hit = json.loads(path.read_text())
        _count(hit.get("usage"), cached=True)
        return hit["content"]
    base = os.environ["VPQA_API_BASE"].rstrip("/")
    headers = {"Authorization": f"Bearer {os.environ['VPQA_API_KEY']}"}
    content, usage = None, None
    payload = {**body, "usage": {"include": True}} if "openrouter" in base else body  # not part of the cache key
    for attempt in range(6):
        try:
            r = httpx.post(f"{base}/chat/completions", json=payload, headers=headers, timeout=120)
            if r.status_code == 429 or r.status_code >= 500:
                raise httpx.HTTPStatusError("retryable", request=r.request, response=r)
            r.raise_for_status()
            content = r.json()["choices"][0]["message"].get("content")
            usage = r.json().get("usage")
            if content:
                break
        except (httpx.TransportError, httpx.HTTPStatusError) as e:
            if attempt == 5 or (isinstance(e, httpx.HTTPStatusError) and e.response.status_code != 429
                                and e.response.status_code < 500):
                raise
        time.sleep(2 ** attempt)
    _count(usage, cached=False)
    if not content:  # empty replies are not cached, so a rerun tries again
        return ""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"request": body, "content": content, "usage": usage}))
    return content


def chat_json(messages: list[dict], **kw) -> dict | None:
    """The first JSON object in the reply, or None if there is none."""
    m = re.search(r"\{.*\}", chat(messages, **kw), re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        return None
