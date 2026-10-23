"""Chat completions through any OpenAI-compatible API, configured by VPQA_API_BASE / VPQA_API_KEY / VPQA_MODEL.

VPQA_MODEL answers (the DAG's answer and revise nodes, the ReAct agent); VPQA_JUDGE_MODEL, when set, does
the judging (the verifier and the evaluation judges), so a cheap answer model can be checked by another one.

Responses are cached on disk by (model, messages, temperature), so reruns cost nothing and are reproducible.
Each cached response keeps the token usage (and, on OpenRouter, the cost) of the call that produced it;
TOTALS sums them over every call in this process, cached or not, so a rerun reports what the run cost.
VPQA_CACHE sets the cache directory; an empty one samples the same run again (the models are not
deterministic at temperature 0). A reply whose reasoning used up max_tokens before any text is asked again
with four times the room, and cached under the original request.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from pathlib import Path

import httpx

CACHE = Path(os.environ.get("VPQA_CACHE", "data/cache/llm"))  # another directory: sample the same run again
_LOCK = threading.Lock()
TOTALS = {"calls": 0, "cached": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0, "cost_by_model": {}}


def _count(model_name: str, usage: dict | None, cached: bool) -> None:
    with _LOCK:  # calls may come from several threads (scripts/eval_answers.py --workers)
        TOTALS["calls"] += 1
        TOTALS["cached"] += cached
        for k in ("prompt_tokens", "completion_tokens", "cost"):
            TOTALS[k] += (usage or {}).get(k) or 0
        by = TOTALS["cost_by_model"]
        by[model_name] = by.get(model_name, 0.0) + ((usage or {}).get("cost") or 0)


def model() -> str:
    return os.environ["VPQA_MODEL"]


def judge_model() -> str:
    return os.environ.get("VPQA_JUDGE_MODEL") or model()


def chat(messages: list[dict], temperature: float = 0.0, max_tokens: int = 800, cache: Path = CACHE,
         model_name: str | None = None) -> str:
    body = {"model": model_name or model(), "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    return complete(body, cache).get("content") or ""


def chat_tools(messages: list[dict], tools: list[dict], tool_choice: str = "auto", temperature: float = 0.0,
               max_tokens: int = 800, cache: Path = CACHE) -> dict:
    """One assistant turn that may call tools: the message, with `content` and possibly `tool_calls`."""
    body = {"model": model(), "messages": messages, "tools": tools, "tool_choice": tool_choice,
            "temperature": temperature, "max_tokens": max_tokens}
    return complete(body, cache)


def complete(body: dict, cache: Path = CACHE) -> dict:
    """The assistant message for a chat/completions request body, cached by the body."""
    key = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    path = cache / key[:2] / f"{key}.json"
    if path.exists():
        hit = json.loads(path.read_text())
        _count(body["model"], hit.get("usage"), cached=True)
        return hit.get("message") or {"role": "assistant", "content": hit["content"]}
    base = os.environ["VPQA_API_BASE"].rstrip("/")
    headers = {"Authorization": f"Bearer {os.environ['VPQA_API_KEY']}"}
    message, usage = None, None
    payload = {**body, "usage": {"include": True}} if "openrouter" in base else body  # not part of the cache key
    for attempt in range(20):  # about 15 minutes: rides out a dropped connection
        try:
            r = httpx.post(f"{base}/chat/completions", json=payload, headers=headers, timeout=90)
            if r.status_code == 429 or r.status_code >= 500:
                raise httpx.HTTPStatusError("retryable", request=r.request, response=r)
            r.raise_for_status()
            choice = r.json()["choices"][0]
            message, usage = choice["message"], r.json().get("usage")
            if message.get("content") or message.get("tool_calls"):
                break
            if choice.get("finish_reason") == "length" and payload["max_tokens"] < 4 * body["max_tokens"]:
                payload = {**payload, "max_tokens": 4 * body["max_tokens"]}  # reasoning used up the reply: more room
                continue
            if choice.get("finish_reason") == "length" or attempt >= 2:
                break
        except (httpx.TransportError, httpx.HTTPStatusError) as e:
            if attempt == 19 or (isinstance(e, httpx.HTTPStatusError) and e.response.status_code != 429
                                and e.response.status_code < 500):
                raise
        time.sleep(min(2 ** attempt, 60))
    _count(body["model"], usage, cached=False)
    if not message or not (message.get("content") or message.get("tool_calls")):  # empty: not cached, retried next run
        return {"role": "assistant", "content": ""}
    message = {k: message[k] for k in ("role", "content", "tool_calls") if message.get(k) is not None}
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {"request": body, "content": message.get("content"), "usage": usage}
    if "tool_calls" in message:
        entry["message"] = message
    tmp = path.with_suffix(f".{threading.get_ident()}.tmp")  # another thread may read the same key
    tmp.write_text(json.dumps(entry))
    tmp.replace(path)
    return message


def chat_json(messages: list[dict], **kw) -> dict | None:
    """The first JSON object in the reply, or None if there is none."""
    m = re.search(r"\{.*\}", chat(messages, **kw), re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        return None


def judge_json(messages: list[dict], **kw) -> dict | None:
    """chat_json on the judge model."""
    return chat_json(messages, model_name=judge_model(), **kw)
