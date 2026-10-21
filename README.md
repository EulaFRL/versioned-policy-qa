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
query ─▶ route (department) ─▶ corpus filter (effective date) ─▶ hybrid retrieve (BM25 + dense, RRF)
      ─▶ rerank ─▶ cited answer ─▶ verifier (format, version, support) ─▶ revise ≤ 2 ─▶ answer | refuse | passages
```

The pipeline is a fixed DAG (`src/vpqa/pipeline/engine.py`, `dag.py`): each node declares typed inputs
and outputs, checked when the graph is built and at run time, and has a time budget and a fallback
(route → all departments, retrieve → BM25 only, rerank → fused order, answer and verify → show the top
passages). The revision loop is unrolled into fixed nodes that are skipped once an answer passes. Every
run writes a JSONL trace: the full inputs, then per node its status, latency and outputs.

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
- Citation accuracy: Cite-Hit, the answer cites a gold passage; Cite-Prec, the share of cited passages
  that are gold
- Version accuracy: on the eCFR version-conflict questions, R@k of the version in force on the question's
  date, plus how often a stale or not-yet-effective version is retrieved instead
- Refusal: precision/recall of refusing, on the same turns made unanswerable by withholding their gold
  documents

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

### Cited answers on MultiDoc2Dial

The answer model sees the top 5 reranked passages (each cut to 400 words) and the last 4 utterances, and
must end every sentence with the passages it relies on, or reply NOT_FOUND. The verifier checks the
format, that every cited version is in force on the question's date, and, in one LLM call, that each
claim is supported by the passages it cites. A rejected answer is rewritten with the verifier's problems,
at most twice; if none passes, the top passages are shown instead. claude-sonnet-5.5 for answer and
verifier, temperature 0.

300 validation turns (seed 0), all answerable:

| | Value |
|---|---|
| Answered / refused / passages shown | 0.833 / 0.167 / 0 |
| Cite-Hit (final answers) | 0.552 |
| Cite-Hit when a gold passage is in the top 5 | 0.868 |
| Gold passage in the top 5 | 0.540 |
| Refused although a gold passage was in the top 5 | 0.019 |
| Cite-Prec | 0.300 |
| First answers rejected by the verifier / passed after one or two rewrites | 25 / 24 + 1 |
| Stage fallbacks | 0 |
| Latency p50 / p95 (end to end) | 6.4 s / 15.4 s |
| API cost | 1.73 USD (0.006 per turn) |

Retrieval is the bottleneck: when a gold passage is shown, the answer cites it 87% of the time; 8 of
the 21 misses cite another passage of the gold document. The verifier changes little here (first-answer
Cite-Hit 0.556): of the 25 rejected answers, 16 had a claim the cited passage does not support (an
overstatement, or a conclusion about the user's case), 8 an uncited sentence, usually a remark on what
the passages leave out, and 1 a verifier reply that could not be parsed; rewriting fixes the wording
but leaves the evidence as it was (10 of the 25 cited a gold passage before the rewrite, 9 after).

### Refusal

Unanswerable turns are made from the same 300: every passage of the documents holding a gold passage is
masked out of retrieval for that turn (`--withhold gold-doc`; masked, not removed, so BM25 statistics do
not change). Withholding a document does not always make a turn unanswerable, because MultiDoc2Dial pages
overlap. An LLM judge given the conversation, the gold grounding text and the top 5 passages retrieved
without the gold document found 109 of 300 withheld turns still answerable (`scripts/judge_withheld.py`,
0.84 USD); those pairs are dropped below.

191 pairs (answerable turn and its withheld twin):

| Declined | P | R | F1 |
|---|---|---|---|
| First answer (no verifier) | 0.664 | 0.466 | 0.548 |
| Final output (verifier, rewrites) | 0.667 | 0.471 | 0.552 |

Of the 191 unanswerable turns, 102 get a first answer, the verifier passes 79 of them unchanged, and 101
end answered. The verifier checks that
each claim is supported by the passages cited, not that the passages answer the question, so an answer
about a neighbouring topic passes.

A refusal gate on the best rerank score (refuse before the LLM call when it is below a threshold) does
not help. On 1,000 train pairs and on validation the score separates answerable from withheld turns only
weakly (AUROC 0.67 and 0.63), and the threshold that maximises the gate's refusal F1 on train (0.996)
leaves only 15% of validation turns answered. The gate is implemented (`answer.min_score`) but off;
`scripts/select_refusal_gate.py` gives the full coverage–accuracy curve.

### Stale versions at answer time (eCFR)

100 dev items, three ways a version not in force on the question's date can reach the answer model:
`filter`, the pipeline as configured (retrieval keeps only versions in force); `unaware`, retrieval over
all versions; `inject`, as `filter`, then another version of the gold paragraph that is not in force is
put first among the 5 passages shown. The answer prompt does not show the question's date; the verifier
checks every cited version against it.

| Versions | First answer cites a version not in force | Of those, rejected | Final cites one | Final Cite-Hit | Answered / refused / passages | Latency p50 / p95 | Cost per item |
|---|---|---|---|---|---|---|---|
| filter | 0 | – | 0 | 0.958 | 0.95 / 0.05 / 0 | 6.2 s / 12.4 s | 0.009 USD |
| unaware | 0.897 | 1.00 | 0.022 | 0.888 | 0.81 / 0.11 / 0.08 | 13.0 s / 27.1 s | 0.018 USD |
| inject | 0.909 | 1.00 | 0.011 | 0.947 | 0.90 / 0.05 / 0.05 | 15.3 s / 28.1 s | 0.019 USD |

Shown a stale version, the model cites it in about 90% of answers, usually next to the right one. The
verifier's date check rejects every one of them, and the rewrite drops the stale citation in all but 1–2%
of items; the price is one or two extra LLM rounds (twice the latency and cost) and 5–8% of items
falling back to passage text. Filtering at retrieval avoids all of that, so the verifier's version check
is a safety net, not the mechanism.

### Fixed DAG vs a ReAct agent

Control group (`src/vpqa/pipeline/react.py`): the same model and the same retrieval and rerank, exposed
as a `search(query, department)` tool; the model decides whether, what and where to search, at most 4
times, then answers under the same citation rules. No verifier inside the loop; afterwards each ReAct
answer is checked once by the DAG's verifier, only to measure support. First 100 of the 300 turns, and
the same 100 with gold documents withheld.

| 100 answerable turns | Answered | Cite-Hit | Cite-Prec | Verifier pass | Searches | Latency p50 / p95 | Cost per turn |
|---|---|---|---|---|---|---|---|
| DAG, final | 0.780 | 0.603 | 0.327 | 0.910 (first answer) | 1 | 6.1 s / 13.7 s | 0.006 USD (with verifier) |
| ReAct | 0.960 | 0.417 | 0.237 | 0.802 | 1.59 | 7.2 s / 12.9 s | 0.009 USD (without) |

| 68 truly unanswerable pairs | Refusal P | R | F1 |
|---|---|---|---|
| DAG, final | 0.642 | 0.500 | 0.562 |
| ReAct | 0.800 | 0.235 | 0.364 |

The agent always searches (never answers from memory), never repeats a query, and when it names a
department (70% of searches) it is always the dialogue's own. Its searches are as good as the DAG's: a
gold passage is among what it sees on 55 of 100 turns, against 52 in the DAG's top 5, although it
rewrites every message into its own query and sees 7.4 passages per turn. The gap is in answering: given
a gold passage it cites it less often (0.77 against 0.94 on answered turns), and it answers almost every
turn, including those where nothing relevant was found, so it refuses half as many unanswerable ones and
its answers pass the verifier less often. The fixed query, the fixed top 5 and the verifier keep the DAG
answer close to its evidence; the agent's freedom to search more does not buy better evidence here.

### Searching each department when the router is unsure

When the router's confidence is below its threshold the pipeline searches all departments in one index.
The alternative (`route.fan_out`, off by default): search each department in its own thread with its own
250 ms budget, fuse BM25 and dense per department, fuse the departments, rerank. Validation, the 439 of
4,201 answerable turns the router leaves unrouted, every list reranked in full:

| Search | R@1 | R@5 | R@10 | MRR@20 | Candidates | Retrieve p50 | Rerank p50 | Departments over budget |
|---|---|---|---|---|---|---|---|---|
| one global search | 0.126 | **0.486** | **0.577** | **0.280** | 35 | 28 ms | 162 ms | – |
| per department, top 5 each | 0.130 | 0.450 | 0.475 | 0.258 | 37 | 99 ms | 182 ms | 10 |
| per department, top 10 each | 0.124 | 0.476 | 0.563 | 0.272 | 73 | 100 ms | 309 ms | 11 |
| gold department only (upper bound) | 0.139 | 0.526 | 0.632 | 0.305 | 33 | 29 ms | 170 ms | – |

Per-department search is slower and no better than one global search, and even searching only the
right department gains just 4 points of R@5 on these turns: what the router misses costs little. It
stays off.

### Stage budgets under pressure

The same 300 turns through the retrieval stages only, with every stage budget (route 150 ms, retrieve
300 ms, rerank 1,500 ms) multiplied by a factor; on CPU, the models moved off the GPU. A stage over
budget falls back (route → all departments, retrieve → BM25, rerank → fused order). A stage that times
out keeps running in its thread and holds the model lock, so the next stage can find the lock taken;
that cost is part of what is measured (`scripts/eval_budgets.py`).

| Device, budget | Route fell back | Retrieve | Rerank | R@1 | R@5 | MRR@20 | Latency p50 / p95 |
|---|---|---|---|---|---|---|---|
| GPU (MPS), no budgets | 0 | 0 | 0 | 0.212 | 0.540 | 0.353 | 278 / 538 ms |
| GPU, × 1 (configured) | 0 | 0 | 0 | 0.212 | 0.540 | 0.353 | 278 / 538 ms |
| GPU, × 0.5 | 0.003 | 0.003 | 0 | 0.212 | 0.540 | 0.353 | 278 / 538 ms |
| GPU, × 0.25 | 0.140 | 0.093 | 0.077 | 0.212 | 0.520 | 0.346 | 277 / 475 ms |
| GPU, × 0.1 | 0.843 | 0.707 | 0.537 | 0.170 | 0.433 | 0.283 | 221 / 308 ms |
| CPU, no budgets | 0 | 0 | 0 | 0.212 | 0.540 | 0.354 | 2,856 / 6,544 ms |
| CPU, × 1 | 0.903 | 0.830 | 0.910 | 0.157 | 0.383 | 0.258 | 1,994 / 2,041 ms |
| CPU, × 0.5 | 1.000 | 0.997 | 1.000 | 0.143 | 0.363 | 0.246 | 1,024 / 1,072 ms |

On the hardware they were set for, the budgets never fire, and they have room: halved, nothing changes;
at a quarter, 8–14% of turns fall back at each stage and R@5 drops 2 points. On CPU, where retrieval
takes 3–6 s, the same budgets hold every turn to about 2 s (their sum is 1.95 s) and nearly every stage
falls back; the ranking then comes from BM25 alone, at R@5 0.38 instead of 0.54. Without budgets the
fallbacks would never run and the CPU p95 would be 6.5 s.

### Replaying a run from its trace

`scripts/replay_trace.py <trace> <run_id>` rebuilds the turn from the trace's first record, runs it
again without budgets (LLM replies come from the response cache, so it costs nothing when no prompt
changed), and compares every node's outputs with the recorded ones: a node marked DIFFERENT is where
code, index or prompt changed since the run; a node that timed out shows what it would have returned.

Reproduce (generation needs VPQA_API_BASE / VPQA_API_KEY / VPQA_MODEL; about 11 USD in all):

```
python scripts/eval_answers.py --dataset md2d --n 300
python scripts/eval_answers.py --dataset md2d --n 300 --withhold gold-doc
python scripts/judge_withheld.py runs/answers/md2d_validation_n300_s0_withheld.jsonl
python scripts/refusal_report.py runs/answers/md2d_validation_n300_s0.jsonl \
    runs/answers/md2d_validation_n300_s0_withheld.jsonl \
    --covered runs/answers/md2d_validation_n300_s0_withheld.covered.json
python scripts/select_refusal_gate.py --train-n 1000
python scripts/eval_answers.py --dataset ecfr --split dev --n 100 --versions filter   # or unaware / inject
python scripts/eval_react.py --n 100                        # and --withhold gold-doc
python scripts/eval_fanout.py --k 5 10
python scripts/eval_budgets.py --n 300                      # and --device cpu --scale none 1 0.5
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
