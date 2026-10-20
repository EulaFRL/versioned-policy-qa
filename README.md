# versioned-policy-qa

Grounded, version-aware question answering over multi-department policy documents.

Most policy/handbook RAG demos answer from whatever chunk ranks first. In real policy QA the hard
failures are different: citing a **superseded version**, mixing rules from **two departments**, and
answering confidently when the documents **don't cover** the question. This project builds a fixed
pipeline (no free-form agent loop) with an explicit evidence verifier, and evaluates each failure mode
separately.

> Status: work in progress. Every number below comes from a run in this repo; stages without numbers
> are not built yet.

## Pipeline

```
query ─▶ rewrite (coreference, aliases) ─▶ route (department) ─▶ hybrid retrieve (BM25 + dense, RRF)
      ─▶ rerank ─▶ version filter (effective date) ─▶ answer ─▶ verifier (claim ↔ span) ─▶ answer | refuse
```

Each stage is a plain function with typed inputs/outputs so it can be ablated on its own.

## Data

| Source | Use | Notes |
|---|---|---|
| [MultiDoc2Dial](https://doc2dial.github.io/multidoc2dial/) | multi-turn, multi-document dialogues with gold grounding spans | 4 domains; used for routing, follow-up rewriting, citation accuracy |
| [eCFR point-in-time API](https://www.ecfr.gov/developers/documentation/api/v1) | multiple dated versions of the same regulation | used to build the version-conflict test split |

Raw data is not committed; see `scripts/download_data.py`.

## Evaluation

- Retrieval: Recall@k, MRR against gold grounding sections
- Citation accuracy (planned): answer cites a span that overlaps gold
- Version accuracy (planned): on the version split, cites the version effective on the query date
- Refusal (planned): precision/recall of refusing on unanswerable turns

### Retrieval on MultiDoc2Dial

Setup: 488 documents split into 13,421 section-level passages (heading path prepended). Each agent
response is one evaluation point: the query is the preceding user utterance, the gold passage is the
section holding the agent's grounding span. Validation split, 4,201 answerable turns. BM25 and dense
(bge-m3) each return top 20; RRF with k = 60.

**Query construction.** Short follow-ups ("yes", "no, it was cancelled") answer the agent rather than ask
something, and cannot be retrieved on their own; longer ones are usually self-contained, and adding
history only dilutes them. So the query is the current utterance alone, plus the last two user/agent
rounds (capped at 96 words, oldest dropped first) only when the utterance has at most 8 words. The
threshold was picked from 3–10 words by RRF Recall@5 on 4,000 sampled train turns
(`scripts/select_gate.py`).

| Retriever | Query | R@1 | R@5 | R@10 | R@20 | MRR@20 |
|---|---|---|---|---|---|---|
| BM25 | current utterance | 0.104 | 0.290 | 0.367 | 0.446 | 0.190 |
| bge-m3 | current utterance | 0.097 | 0.318 | 0.403 | 0.484 | 0.193 |
| RRF | current utterance | 0.132 | 0.356 | 0.430 | 0.502 | 0.231 |
| BM25 | length-gated history | 0.135 | 0.362 | 0.448 | 0.537 | 0.238 |
| bge-m3 | length-gated history | 0.137 | 0.403 | 0.499 | 0.591 | 0.253 |
| **RRF** | **length-gated history** | **0.166** | **0.434** | **0.527** | **0.603** | **0.284** |
| RRF | oracle gate (gold dialogue act; upper bound) | 0.173 | 0.443 | 0.529 | 0.612 | 0.292 |

Single-query latency on a laptop (Apple M5 Pro): BM25 30 ms, bge-m3 24 ms with the gated query.

Reproduce: `python scripts/eval_retrieval.py --split validation --query-mode last gated oracle`.

## Layout

```
src/vpqa/
  data/        dataset loaders and normalization to a common Turn/Passage schema
  retrieval/   bm25, dense (bge-m3), RRF fusion, cross-encoder rerank
  pipeline/    the fixed DAG and the verifier
  eval/        metrics and the eval runner
scripts/       data download, index build, eval entry points
configs/       run configs
tests/         unit tests for metrics and fusion
```

## Setup

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -e ".[dev]"
python scripts/download_data.py
pytest
```

Everything runs on a laptop: embeddings and reranker are local, generation goes through an
OpenAI-compatible API set by `VPQA_API_BASE` / `VPQA_API_KEY` / `VPQA_MODEL`.
