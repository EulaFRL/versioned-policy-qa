"""Chat completions through any OpenAI-compatible API, configured by VPQA_API_BASE / VPQA_API_KEY / VPQA_MODEL.

Responses are cached on disk by (model, messages, temperature), so reruns cost nothing and are reproducible.
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


def model() -> str:
    return os.environ["VPQA_MODEL"]


def chat(messages: list[dict], temperature: float = 0.0, max_tokens: int = 800, cache: Path = CACHE) -> str:
    body = {"model": model(), "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    key = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    path = cache / key[:2] / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())["content"]
    base = os.environ["VPQA_API_BASE"].rstrip("/")
    headers = {"Authorization": f"Bearer {os.environ['VPQA_API_KEY']}"}
    content = None
    for attempt in range(6):
        try:
            r = httpx.post(f"{base}/chat/completions", json=body, headers=headers, timeout=120)
            if r.status_code == 429 or r.status_code >= 500:
                raise httpx.HTTPStatusError("retryable", request=r.request, response=r)
            r.raise_for_status()
            content = r.json()["choices"][0]["message"].get("content")
            if content:
                break
        except (httpx.TransportError, httpx.HTTPStatusError) as e:
            if attempt == 5 or (isinstance(e, httpx.HTTPStatusError) and e.response.status_code != 429
                                and e.response.status_code < 500):
                raise
        time.sleep(2 ** attempt)
    if not content:  # empty replies are not cached, so a rerun tries again
        return ""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"request": body, "content": content}))
    return content


def chat_json(messages: list[dict], **kw) -> dict | None:
    """The first JSON object in the reply, or None if there is none."""
    m = re.search(r"\{.*\}", chat(messages, **kw), re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        return None
