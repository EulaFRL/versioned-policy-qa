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
| [eCFR point-in-time API](https://www.ecfr.gov/developers/documentation/api/v1) | multiple dated versions of the same regulation | 20 CFR 404/416, 38 CFR 3, 34 CFR 668/685 (the regulations behind three MultiDoc2Dial domains); used to build the version-conflict questions |

Raw data is not committed; see `scripts/download_data.py` and `scripts/build_ecfr.py`. The eCFR
version-conflict questions are committed in `data/benchmark/`: `ecfr_version_qa.jsonl` holds the items,
`ecfr_version_qa.log.jsonl` the outcome of every candidate pair, including the rejected ones.

## Evaluation

- Retrieval: Recall@k, MRR against gold grounding sections; Hit@k and Ctx@W (gold text within a word
  budget) when comparing retrieval units
- Citation accuracy (planned): answer cites a span that overlaps gold
- Version accuracy: on the eCFR version-conflict questions, R@k of the version in force on the question's
  date, plus how often a stale or not-yet-effective version is retrieved instead
- Refusal (planned): precision/recall of refusing on unanswerable turns

### Retrieval on MultiDoc2Dial

Setup: 488 documents split into 13,421 section-level passages (heading path prepended). Each agent
response is one evaluation point: the query is the preceding user utterance, the gold passage is the
section holding the agent's grounding span. Validation split, 4,201 answerable turns. BM25 and dense
(bge-m3) each return top 20; RRF with k = 60. 1,714 passages repeat another passage's text word for
word; equal scores are broken by passage order so runs are deterministic.

**Query construction.** Short follow-ups ("yes", "no, it was cancelled") answer the agent rather than ask
something, and cannot be retrieved on their own; longer ones are usually self-contained, and adding
history only dilutes them. So the query is the current utterance alone, plus the last two user/agent
rounds (capped at 96 words, oldest dropped first) only when the utterance has at most 7 words. The
threshold was picked from 3–10 words by RRF Recall@5 on 4,000 sampled train turns
(`scripts/select_gate.py`).

| Retriever | Query | R@1 | R@5 | R@10 | R@20 | MRR@20 |
|---|---|---|---|---|---|---|
| BM25 | current utterance | 0.104 | 0.290 | 0.367 | 0.446 | 0.190 |
| bge-m3 | current utterance | 0.092 | 0.316 | 0.401 | 0.483 | 0.190 |
| RRF | current utterance | 0.131 | 0.353 | 0.429 | 0.502 | 0.229 |
| BM25 | length-gated history | 0.134 | 0.363 | 0.446 | 0.536 | 0.237 |
| bge-m3 | length-gated history | 0.129 | 0.396 | 0.492 | 0.583 | 0.246 |
| **RRF** | **length-gated history** | **0.166** | **0.430** | **0.521** | **0.598** | **0.283** |
| RRF | oracle gate (gold dialogue act; upper bound) | 0.173 | 0.440 | 0.528 | 0.611 | 0.291 |

**Rerank.** bge-reranker-v2-m3 (cross-encoder, fp16 on MPS) reorders the top N of the RRF list; the rest
of the list stays in place. The cross-encoder gets the same gated query.

| Query | Reranked | R@1 | R@5 | R@10 | R@20 | MRR@20 |
|---|---|---|---|---|---|---|
| gated | none (RRF) | 0.166 | 0.430 | 0.521 | 0.598 | 0.283 |
| gated | top 10 | 0.206 | 0.483 | 0.521 | 0.598 | 0.329 |
| gated | top 20 | 0.211 | 0.509 | 0.570 | 0.598 | 0.341 |
| **gated** | **all candidates (≤ 40)** | **0.215** | **0.531** | **0.611** | **0.654** | **0.354** |
| gated, cross-encoder always given history | all | 0.209 | 0.521 | 0.607 | 0.653 | 0.344 |
| current utterance | all | 0.180 | 0.453 | 0.519 | 0.558 | 0.299 |

Reranking the whole candidate list is best on every metric, and it also lifts R@20 because RRF ranks some
gold passages below 20. The cross-encoder does not remove the need for the gate: giving it the last two
rounds for every turn lowers R@5 on questions from 0.525 to 0.510.

**Department routing.** A MultiDoc2Dial dialogue stays in one domain, so the router reads the whole
dialogue so far (current utterance first, oldest dropped past 256 words); the current utterance alone
routes only 74.8% of held-out train turns correctly, against 97.9% for the dialogue. The router is
logistic regression on bge-m3 embeddings (`scripts/train_router.py`). When its confidence clears a
threshold, BM25 and dense search only that domain; otherwise everything. Thresholds were picked on 20% of
train dialogues held out from fitting, separately for first turns (0.9: one utterance is little to go
on, and a wrong domain removes the gold passage outright) and follow-ups (0.7).

| Routing | RRF R@5 | + rerank R@1 | R@5 | R@10 | R@20 | MRR@20 | first-turn R@5 |
|---|---|---|---|---|---|---|---|
| none | 0.430 | 0.215 | 0.531 | 0.611 | 0.654 | 0.354 | 0.536 |
| **router** | **0.449** | **0.224** | **0.549** | **0.632** | **0.681** | **0.368** | 0.532 |
| gold domain (upper bound) | 0.456 | 0.227 | 0.558 | 0.642 | 0.692 | 0.373 | 0.558 |

On validation the router restricts 89.6% of turns (60.2% of first turns) and picks the right domain for
99.3% of them.

**Retrieval unit.** Sections are small (median 30 words). The MultiDoc2Dial paper uses two coarser units,
rebuilt here from its released preprocessing code with identical passage counts: `heading` merges
consecutive spans that share a section or a title (4,110 passages, mean 112 words), `token` cuts the
document text every 100 tokens (4,283 passages, title prepended). A coarser unit holds more text, so
the gold passage enters the top k more easily; Hit@k (any gold passage in the top k) therefore cannot
compare units. Ctx@W can: the share of gold grounding spans inside the first W words of retrieved text
(whole passages in rank order), i.e. at the same reader budget. Gated query, no routing:

| Unit | RRF Hit@5 | + rerank Hit@1 | Hit@5 | Hit@10 | RRF Ctx@200 | + rerank Ctx@200 | Ctx@500 |
|---|---|---|---|---|---|---|---|
| **section** | 0.432 | 0.217 | 0.533 | 0.612 | **0.393** | **0.473** | 0.611 |
| heading | 0.613 | 0.415 | 0.700 | 0.768 | 0.321 | 0.375 | 0.599 |
| token (100) | 0.592 | 0.408 | 0.674 | 0.740 | 0.316 | 0.409 | **0.637** |

Hit@5 favours coarse units by 14–18 points, but at a 200-word budget sections carry the most gold text:
a few short sections fit where a coarse passage spends the budget on its neighbours. At 500 words the
100-token windows are slightly ahead. The unit stays `section`; which unit is best depends on how much
retrieved text the reader gets, which is fixed with the generator.

**Comparison with the MultiDoc2Dial paper** (Feng et al., 2021, Table 5; validation, R@k there means
Hit@k here). The paper's query is the current turn followed by the whole history, most recent first;
here that is the `history` query mode (96 words).

| Unit | Retriever | Query | Hit@1 | Hit@5 | Hit@10 |
|---|---|---|---|---|---|
| heading | BM25 (paper) | history | 0.196 | 0.419 | 0.508 |
| heading | BM25 | history | 0.227 | 0.472 | 0.558 |
| heading | DPR, NQ-trained (paper) | history | 0.155 | 0.353 | 0.456 |
| heading | bge-m3 | history | 0.240 | 0.510 | 0.619 |
| heading | DPR fine-tuned on MultiDoc2Dial (paper) | history | 0.490 | 0.723 | 0.800 |
| heading | RRF | gated | 0.342 | 0.613 | 0.698 |
| heading | RRF + rerank all | gated | 0.415 | 0.700 | 0.768 |
| token | BM25 (paper) | history | 0.195 | 0.427 | 0.514 |
| token | BM25 | history | 0.226 | 0.470 | 0.560 |

BM25 lands 3–5 points above the paper with the same units and query style (tokenisation and the
96-word cap differ). The gated query alone is worth 9 points of Hit@5 over the paper's query on the
same units (RRF 0.523 → 0.613). Nothing here is trained on MultiDoc2Dial passages, and the paper's
fine-tuned DPR is still ahead, by 7.5 points at Hit@1 and 2.3 at Hit@5 against gated RRF + rerank.

**Latency** for one query at a time on a laptop (Apple M5 Pro), gated query: BM25 17 ms, bge-m3 17 ms,
router 25 ms, rerank of 40 candidates about 250 ms (55 ms for 10, 128 ms for 20). Rerank is the most
expensive retrieval stage.

Reproduce:

```
python scripts/select_gate.py --split train
python scripts/train_router.py
python scripts/eval_retrieval.py --query-mode last gated oracle --rerank 10 20 40
python scripts/eval_retrieval.py --query-mode gated --rerank 40 --rerank-query recent:4
python scripts/eval_retrieval.py --query-mode gated --rerank 40 --route router   # or --route oracle
python scripts/eval_retrieval.py --unit heading --query-mode history                # or --unit token / section
python scripts/eval_retrieval.py --unit heading --query-mode gated --rerank 10 40
```

### Version-aware retrieval on eCFR

**Corpus.** For each of the five parts, eCFR's point-in-time API is queried on every date a section
changed in substance, from 2016-12-19 (the earliest date it serves) to 2026-08-31: 215 snapshots. Each
section is cut at its top-level paragraphs (a), (b), ...; paragraphs over 250 words are cut further at
(1), (2), ... or, for definition lists, at each definition. A passage is keyed by its labels or the
definition's first words, never by position, so inserting a paragraph does not turn the ones after it
into new versions. Consecutive snapshots with the same text under a key form one version, valid from the
first snapshot date until the date the text changes; changes in typesetting only (dash and spacing
re-renders) do not count. Result: 12,483 passage versions, 1,437 paragraphs with more than one.

**Questions.** From each amended paragraph (at least 25 words, at least 3 changed words) one pair of
consecutive versions is sampled, 1,012 pairs in all. An LLM first checks that the two texts are the same
provision: when paragraphs are relettered, one label can hold unrelated rules over time. It then writes
a question that both versions answer explicitly but differently, or declines. The question is kept only
if each version, given alone, answers it and a judge finds the two answers materially different.
Outcomes: different provision 165, declined 324, kept 277, a version could not answer 165, same answer
52, malformed generator output 29 (discarded). Each kept question gives two items, dated inside each
version's validity window, whose gold passage is the version in force on that date. Splits are by
section: dev 135 questions (270 items, 89 sections), test 142 questions (284 items, 84 sections). All
LLM steps use claude-sonnet-5.5 at temperature 0; the set cost about 9 USD including pilot runs. The
test split is held out until the end of the project.

**Spot check.** 30 questions drawn at random before the same-provision check was added were read against
both texts. The check removes the 3 that compare unrelated rules. Of the other 27, two questions give
the change away in their wording and one answer adds text both versions share; none has a wrong gold
version.

**Policies.** `unaware`: one index over all versions, date ignored. `latest`: only versions in force
today. `prefilter`: retrieve among versions in force on the question's date. `postfilter:N`: retrieve
top N over all versions, then drop those not in force. BM25 and bge-m3 each return 20 (or N), fused
with RRF and reranked by bge-reranker-v2-m3. Key@5: any version of the gold paragraph in the top 5.
Stale@1: the top passage is another version of the gold paragraph. Invalid@5: share of the top 5 not
in force on the question's date.

Dev split, 270 items, RRF + rerank:

| Policy | R@1 | R@5 | R@10 | Key@5 | Stale@1 | Invalid@5 | R@1 old | R@1 new |
|---|---|---|---|---|---|---|---|---|
| unaware | 0.293 | 0.819 | 0.900 | 0.911 | 0.293 | 0.505 | 0.267 | 0.319 |
| latest | 0.248 | 0.407 | 0.419 | 0.889 | 0.293 | 0.399 | 0.000 | 0.496 |
| prefilter | **0.581** | 0.915 | 0.937 | 0.915 | 0 | 0 | 0.600 | 0.563 |
| postfilter:20 | 0.578 | 0.911 | 0.926 | 0.911 | 0 | 0 | 0.600 | 0.556 |
| prefilter, depth 100 | **0.581** | **0.937** | **0.959** | 0.937 | 0 | 0 | 0.600 | 0.563 |
| postfilter:100 | **0.581** | **0.937** | **0.959** | 0.937 | 0 | 0 | 0.600 | 0.563 |

Without the date the retriever still finds the right paragraph (Key@5 0.911) but picks the wrong
version as often as the right one: Stale@1 equals R@1. Filtering by date doubles R@1. Keeping only the
current text fails every question dated in the past and also loses on many "new" questions, because
that version was itself amended later. Filtering before or after retrieval makes no difference at equal
depth, since only two or three versions of a paragraph compete; postfilter:20 loses a little at R@10
because some of its 20 candidates are dropped. Without the reranker, prefilter reaches R@1 0.433 and
unaware 0.200.

Reproduce:

```
python scripts/build_ecfr.py
python scripts/make_version_qa.py          # needs VPQA_API_BASE / VPQA_API_KEY / VPQA_MODEL
python scripts/eval_versions.py --split dev --rerank \
    --policy unaware latest prefilter postfilter:20 prefilter:100 postfilter:100
```

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
