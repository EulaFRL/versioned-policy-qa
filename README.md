# versioned-policy-qa

Grounded, version-aware question answering over multi-department policy documents.

Most policy/handbook RAG demos answer from whatever chunk ranks first. In real policy QA the hard
failures are different: citing a **superseded version**, mixing rules from **two departments**, and
answering confidently when the documents **don't cover** the question. This project builds a fixed
pipeline (no free-form agent loop) with an explicit evidence verifier, and evaluates each failure mode
separately.

> Status: early work in progress. Numbers below will be filled in only from runs in this repo.

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

## Evaluation (planned)

- Retrieval: Recall@k, MRR against gold grounding spans
- Citation accuracy: answer cites a span that overlaps gold
- Version accuracy: on the version split, cites the version effective on the query date
- Refusal: precision/recall of refusing on unanswerable turns

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
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python scripts/download_data.py
pytest
```

Everything runs on a laptop: embeddings and reranker are local, generation goes through an
OpenAI-compatible API set by `VPQA_API_BASE` / `VPQA_API_KEY` / `VPQA_MODEL`.
