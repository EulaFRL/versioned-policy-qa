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
`ecfr_version_qa.log.jsonl` the outcome of every candidate pair, including the rejected ones. So are the
near-miss unanswerable turns built from MultiDoc2Dial (`md2d_near_miss_{train,validation}.jsonl`, see
[Refusal](#refusal)), every rewrite with the checker's verdict, kept or not.

## Evaluation

- Retrieval: Recall@k, MRR against gold grounding sections; Hit@k and Ctx@W (gold text within a word
  budget) when comparing retrieval units
- Citation accuracy: Cite-Hit, the answer cites a gold passage; Cite-Prec, the share of cited passages
  that are gold
- Version accuracy: on the eCFR version-conflict questions, R@k of the version in force on the question's
  date, plus how often a stale or not-yet-effective version is retrieved instead
- Refusal: precision/recall of refusing, on the same turns made unanswerable two ways: their gold
  documents withheld, or the question rewritten to ask what the documents do not say (near-miss)

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
at most twice; if none passes, the top passages are shown instead. Temperature 0. Two answer models:
claude-sonnet-5.5, and deepseek-v4-flash at 1/22 of its input and 1/55 of its output token price; the verifier is
claude-sonnet-5.5 in both (`VPQA_MODEL` answers, `VPQA_JUDGE_MODEL` judges).

300 validation turns (seed 0), all answerable:

| | Sonnet 5.5 | DeepSeek V4 Flash |
|---|---|---|
| Answered / refused / passages shown | 0.833 / 0.167 / 0 | 0.683 / 0.293 / 0.023 |
| Cite-Hit (final answers) | 0.552 | 0.547 |
| Cite-Hit when a gold passage is in the top 5 | 0.868 | 0.823 |
| Gold passage in the top 5 | 0.540 | 0.540 |
| Refused although a gold passage was in the top 5 | 0.019 | 0.130 |
| Cite-Prec | 0.300 | 0.320 |
| First answers rejected by the verifier / passed after one or two rewrites | 25 / 24 + 1 | 34 / 18 + 0 |
| Stage fallbacks | 0 | 0.03 (answer or revise over 60 s: 6; route: 3) |
| Latency p50 / p95 (end to end) | 6.4 s / 15.4 s | see [eCFR](#stale-versions-at-answer-time-ecfr) |
| API cost per turn (answer model / all) | – / 0.006 USD | 0.0001 / 0.0014 USD |

Retrieval is the bottleneck: when a gold passage is shown, Sonnet cites it 87% of the time; 8 of
the 21 misses cite another passage of the gold document. The verifier changes little here (first-answer
Cite-Hit 0.556): of the 25 rejected answers, 16 had a claim the cited passage does not support (an
overstatement, or a conclusion about the user's case), 8 an uncited sentence, usually a remark on what
the passages leave out, and 1 a verifier reply that could not be parsed; rewriting fixes the wording
but leaves the evidence as it was (10 of the 25 cited a gold passage before the rewrite, 9 after).

DeepSeek cites as accurately when it answers, at a quarter of the cost per turn (the verifier is now most
of it), but it refuses more: 13% of turns with a gold passage shown, against 2%. Its rejected answers are
fixed less often (18 of 34 after one rewrite; 9 end refused, 7 as passage text). The DeepSeek run reused
answer-model replies cached by an earlier run that a network outage cut short, so its latency is not
reported here; the eCFR run below has fresh calls throughout.

### Refusal

Unanswerable turns are made from the same 300: every passage of the documents holding a gold passage is
masked out of retrieval for that turn (`--withhold gold-doc`; masked, not removed, so BM25 statistics do
not change). Withholding a document does not always make a turn unanswerable, because MultiDoc2Dial pages
overlap. An LLM judge given the conversation, the gold grounding text and the top 5 passages retrieved
without the gold document found 109 of 300 withheld turns still answerable (`scripts/judge_withheld.py`,
0.84 USD); those pairs are dropped below.

191 pairs (answerable turn and its withheld twin):

| Declined, P / R / F1 | Sonnet 5.5 | DeepSeek V4 Flash |
|---|---|---|
| First answer (no verifier) | 0.664 / 0.466 / 0.548 | 0.618 / 0.534 / 0.573 |
| Final output (verifier, rewrites) | 0.667 / 0.471 / 0.552 | 0.605 / 0.602 / 0.604 |

With Sonnet answering, 102 of the 191 unanswerable turns get a first answer, the verifier passes 79 of
them unchanged, and 101 end answered. The verifier checks that each claim is supported by the passages
cited, not that the passages answer the question, so an answer about a neighbouring topic passes.

**Near-miss turns.** Withholding a document tests evidence that is missing altogether. The harder case
is retrieval returning the right page when that page answers a neighbouring question. For each of the
300 turns (and 200 train turns, seed 1, for development), `scripts/make_near_miss.py` has
claude-sonnet-5.5 rewrite the latest user message, given the conversation and the gold passages, to
ask about one condition, group, amount or detail the text does not state. A checker from another model
family (gpt-5-mini) is shown the whole gold document and the top 5 passages retrieved for the rewrite
and says whether they answer it; only the rewrites it calls unanswered are kept: 174 of 300 (117 of 200
on train; 1.08 USD for validation). The set was fixed before any system was run on it.

174 pairs (answerable turn and its near-miss rewrite):

| Declined, P / R / F1 | Sonnet 5.5 | DeepSeek V4 Flash |
|---|---|---|
| First answer (no verifier) | 0.724 / 0.408 / 0.522 | 0.742 / 0.678 / 0.709 |
| Final output (verifier, rewrites) | 0.724 / 0.408 / 0.522 | 0.737 / 0.787 / 0.761 |

Near-miss rewrites are refused more often than withheld turns, although a passage of the gold page is in
the top 5 for 57% of them (54% of the original turns): shown the page, the model can see what it leaves
out. The models differ: Sonnet answers 103 of 174; the verifier rejects 46 of those, and every
rewrite becomes an answer that passes. DeepSeek answers 56; the verifier rejects 30, and 19 of those are
rewritten to NOT_FOUND. Many answers counted as not declined are not wrong: they give what the page does say and state that it does not cover the asked detail, or ask
the user to clarify.

**Checking that the answer replies to the question.** The verifier can also judge, in a second call
made only when every claim is supported, whether the answer as a whole replies to the latest user
message (`answer.check_question`, off by default); a failure goes back to the rewrite like any other
problem, and the rewrite may reply NOT_FOUND. It was developed on the train turns (a first version that
asked both questions in one call changed the support verdicts of 26 of 154 answers, so it became a
separate call) and then run once on validation, DeepSeek answering:

| Set | Turns | Measure | Without check | With check | Only without / only with | McNemar p |
|---|---|---|---|---|---|---|
| answerable | 300 | declined | 0.317 | 0.317 | 2 / 2 | 1 |
| answerable | 300 | answered with a gold citation | 0.387 | 0.387 | 1 / 1 | 1 |
| withheld | 191 | declined | 0.602 | 0.623 | 0 / 4 | 0.125 |
| near-miss | 174 | declined | 0.787 | 0.793 | 1 / 2 | 1 |

No difference on any set (train: near-miss declined 0.880 against 0.889, answerable 0.250 against
0.275), at about twice the verifier cost on answered turns. Most answers it could catch are already
refused by the answer model or rejected by the support check. It stays off (`scripts/ab_verifier.py`).

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
| filter, DeepSeek V4 Flash answering | 0 | – | 0 | 0.922 | 0.81 / 0.10 / 0.09 | 8.8 s / 60.6 s | 0.003 USD |

Shown a stale version, the model cites it in about 90% of answers, usually next to the right one. The
verifier's date check rejects every one of them, and the rewrite drops the stale citation in all but 1–2%
of items; the price is one or two extra LLM rounds (twice the latency and cost) and 5–8% of items
falling back to passage text. Filtering at retrieval avoids all of that, so the verifier's version check
is a safety net, not the mechanism.

With DeepSeek answering (verifier still Sonnet), the filtered pipeline costs a third as much per item but
is slower in the tail: its answer call has a p50 of 5.2 s against 2.9 s, and 5 of 100 answer calls and 1
rewrite ran past the 60 s budget and fell back to passage text; with 3 items rejected twice, that makes
the 9% shown as passages.

### Reusing verified answers across dates (eCFR)

An organisation asks the same questions again, so a verified answer is worth keeping
(`src/vpqa/memory.py`, `AnswerMemory`): a later question whose bge-m3 similarity to a stored one is at
least τ gets the stored answer and its citations without running the pipeline. The risk is the one this
project is about: the stored answer cites the versions in force when it was written. `bound` reuses an
entry only if every passage it cites is in force on the new question's date, and otherwise runs the
pipeline; `unbound` reuses on similarity alone. Answers without citations (clarifying questions) are not
stored.

Dev, the 135 eCFR pairs whose two items ask the same question (`scripts/eval_answer_memory.py`). The
memory is filled by running the pipeline (DeepSeek V4 Flash answering, Sonnet verifying) on each question
at its old date; 111 answers come out verified and cited (fill Cite-Hit 0.915). Each question is then
asked again, paraphrased by the answer model (similarity to the original: p10 0.894, p50 0.966), at two
dates: `same window`, another date on which the old gold version is still in force, where reuse is right;
and `new version`, the pair's new date, where reuse cites the version that was replaced. Cite-Hit and
Stale (a final citation not in force on the date) are over turns not refused.

| Memory, τ 0.9 | Question date | Reused | Blocked as stale | Refused | Cite-Hit | Stale |
|---|---|---|---|---|---|---|
| none | same window | 0 | – | 0.259 | 0.850 | 0 |
| unbound | same window | 0.719 | – | 0.141 | 0.897 | 0.026 |
| bound | same window | 0.696 | 0.022 | 0.141 | 0.888 | 0 |
| none | new version | 0 | – | 0.267 | 0.889 | 0 |
| unbound | new version | 0.719 | – | 0.111 | 0.167 | 0.808 |
| bound | new version | 0 | 0.719 | 0.267 | 0.889 | 0 |

| Unbound, new version: Stale by τ | 0.8 | 0.85 | 0.9 | 0.95 |
|---|---|---|---|---|
| Reused | 0.837 | 0.785 | 0.719 | 0.533 |
| Reused from another pair's question | 0.037 | 0.007 | 0 | 0 |
| Stale | 0.883 | 0.848 | 0.808 | 0.632 |

Within the window, reuse answers 7 in 10 questions with no LLM call and refuses fewer of them than a fresh
run (0.14 against 0.26): the memory holds only answers the pipeline gave, and it answered the original
wording more often than the paraphrase (0.82 against 0.74). After the regulation changed, reuse on similarity alone gives the
replaced version's answer to 81% of answered questions; raising τ only trades reuse for staleness, since
the question is the same and only its date differs. Binding the entry to the versions it cites removes
every stale reuse at every τ and returns the no-memory result on new-version questions; in the window it
also blocks the 2% of entries that cite a second passage which changed in the meantime, the case in which
unbound reuse is stale even inside the gold version's window. τ 0.9 was chosen on these same items, so
the full grid is in `runs/answer_memory/ecfr_dev.json`; the test split is not used. About 1 USD.

### Fixed DAG vs a ReAct agent

Control group (`src/vpqa/pipeline/react.py`): the same model and the same retrieval and rerank, exposed
as a `search(query, department)` tool; the model decides whether, what and where to search, at most 4
times, then answers under the same citation rules. No verifier inside the loop; afterwards each ReAct
answer is checked once by the DAG's verifier, only to measure support. First 100 of the 300 turns, and
the same 100 with gold documents withheld; both answer models, Claude Sonnet 5.5 as the verifier. Cite-Hit
and Cite-Prec here are over answered turns only (passage text shown instead of an answer does not count).

| 100 answerable turns | Answered | Cite-Hit | Cite-Prec | Verifier pass | Searches | Latency p50 / p95 | Cost per turn |
|---|---|---|---|---|---|---|---|
| Sonnet, DAG final | 0.780 | 0.603 | 0.327 | 0.910 (first answer) | 1 | 6.1 s / 13.7 s | 0.006 USD (with verifier) |
| Sonnet, ReAct | 0.960 | 0.417 | 0.237 | 0.802 | 1.59 | 7.2 s / 12.9 s | 0.009 USD (without) |
| DeepSeek, DAG first answer | 0.700 | 0.500 | 0.312 | 0.886 | 1 | – | – |
| DeepSeek, DAG final | 0.650 | 0.538 | 0.335 | – | 1 | – | 0.0014 USD (with verifier) |
| DeepSeek, ReAct | 0.980 | 0.449 | 0.250 | 0.694 | 2.24 | 7.2 s / 19.5 s | 0.0003 USD (without) |

DeepSeek DAG latency is not given for the reason above (cached replies); its cost is the 300-turn mean.

| 68 truly unanswerable pairs | Refusal P | R | F1 |
|---|---|---|---|
| Sonnet, DAG final | 0.642 | 0.500 | 0.562 |
| Sonnet, ReAct | 0.800 | 0.235 | 0.364 |
| DeepSeek, DAG final | 0.597 | 0.632 | 0.614 |
| DeepSeek, ReAct | 0.933 | 0.206 | 0.337 |

The agent always searches (never answers from memory) and when it names a department it is almost always
the dialogue's own. With Sonnet its searches are as good as the DAG's: a gold passage is among what it
sees on 55 of 100 turns, against 52 in the DAG's top 5, although it rewrites every message into its own
query and sees 7.4 passages per turn. The gap is in answering: given a gold passage it cites it less
often (0.77 against 0.94 on answered turns), and it answers almost every turn, including those where
nothing relevant was found, so it refuses far fewer unanswerable ones and its answers pass the verifier
less often. The fixed query, the fixed top 5 and the verifier keep the DAG answer close to its evidence;
the agent's freedom to search more does not buy better evidence here.

The cheaper model widens the gap. As an agent it searches more (2.24 searches per turn; 22% of turns
use all 4, 43% with gold withheld) and still answers 98% of turns, 85% with gold withheld; its answers
pass the verifier less often than Sonnet's (0.694 against 0.802). In the DAG the same model is held
back by the verifier: it answers fewer turns than Sonnet, but refuses more of the unanswerable ones
(F1 0.614, the best of the four).

On eCFR (dev, 100 items, every item dated) the agent's search tool takes an `as_of` date and searches
only the versions in force then; the DAG filters by the item's date itself.

| eCFR dev, 100 items | Answered | Cite-Hit | Cite-Prec | Verifier pass | Searches | Date set / right | Cites stale | Latency p50 / p95 |
|---|---|---|---|---|---|---|---|---|
| Sonnet, DAG final | 0.950 | 0.958 | 0.645 | 0.905 (first answer) | 1 | (filter) | 0 | 6.2 s / 12.4 s |
| Sonnet, ReAct | 0.990 | 0.889 | 0.642 | 0.838 | 1.22 | 1.000 / 0.992 | 0.010 | 6.0 s / 11.4 s |
| DeepSeek, DAG final | 0.810 | 0.938 | 0.655 | 0.785 (first answer) | 1 | (filter) | 0 | 8.8 s / 60.6 s |
| DeepSeek, ReAct | 0.990 | 0.808 | 0.613 | 0.717 | 2.18 | 1.000 / 0.991 | 0 | 13.6 s / 44.6 s |

Both agents pass a date on every search and the right one on 99% of them: told the date in the prompt,
they use the tool's filter. Sonnet's agent cites a stale version once in 100 answers, on the search
whose date it got wrong. With the date handled, the result repeats MultiDoc2Dial: the agent answers
more, cites gold less often, and passes the verifier less often. Cost per item: Sonnet ReAct 0.012 USD,
DeepSeek ReAct 0.0006 USD (verifier 0.003 and 0.002 USD on top).

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

### Session memory and query rewriting

A follow-up such as "and if I'm under 18?" names nothing, and a short follow-up's query only adds the
last few utterances (the length gate). Two fixes, measured on retrieval (`scripts/eval_memory.py`):

- **Session focus** (`memory.focus`, `src/vpqa/memory.py`): the session keeps the documents the last
  `replies` replies cited. The best `k` passages of those documents (RRF of BM25 and dense) join the
  rerank candidates, and every passage of a focus document gets `bonus` added to its rerank score. The
  focus is a hint for ranking only; an answer still cites passages it was shown.
- **LLM rewrite** (`query.rewrite`): the answer model (DeepSeek V4 Flash) rewrites the follow-up into a
  standalone question, either only the short follow-ups the gate gives history (`short`) or every
  follow-up (`all`).

Each dialogue is replayed turn by turn. The document of a turn's top-ranked passage stands in for what
its reply would cite (`system`). `oracle` uses the previous turn's gold document instead, an upper
bound. The settings were picked on 200 train dialogues (1,224 answerable turns; replies 1–2, k 0/10,
bonus 0–1). Train R@5: 0.571 without either fix, 0.580 with `short`, 0.583 with `all`, 0.621 with session
focus, 0.628 with both. Validation, all 661 dialogues, 4,201 answerable turns:

| Query | Session | R@1 | R@5 | MRR@20 | Follow-up, same document: R@5 | Follow-up, switches document: R@5 |
|---|---|---|---|---|---|---|
| gated | none | 0.224 | 0.549 | 0.368 | 0.539 | 0.591 |
| gated | bonus only (k 0) | 0.246 | 0.567 | 0.384 | 0.568 | 0.591 |
| gated | focus passages only (bonus 0) | 0.229 | 0.571 | 0.381 | 0.572 | 0.595 |
| gated | focus (k 10, bonus 0.1) | 0.254 | **0.594** | 0.403 | 0.614 | 0.580 |
| LLM rewrite (short) | none | 0.224 | 0.557 | 0.371 | 0.552 | 0.593 |
| LLM rewrite (short) | focus | 0.261 | **0.603** | 0.410 | 0.627 | 0.582 |
| gated | oracle focus | 0.310 | 0.644 | 0.456 | 0.702 | 0.560 |

First turns (660) are 0.532 in every row. Follow-ups that stay on the previous turn's document (2,635)
gain 7.5 points from session focus. Follow-ups that switch document (906) lose 1, because the bonus pulls
toward the old document; the oracle, which is sure of the old document, loses 3 on them. The two parts
add up: extra candidates or the bonus alone give about 2 points each, together 4.5. The LLM rewrite gives
0.8 points R@5 on its own and 0.9 on top of the session, at 6.4 s p50 and 14.9 s p95 per rewritten turn,
more than the whole retrieval budget, for 0.21 USD on validation. It stays off (`query.rewrite: "off"`).

Session focus also stays off, despite the gain in recall. In the offline loop below, the same settings
put gold in the top 5 on 12 more of 200 train turns, but end-to-end success does not rise.

### Offline improvement loop

`scripts/loop.py` (with `src/vpqa/loop.py`) changes the configuration only when a paired test says the
change helps. Every failure on two frozen sets is put into one of four categories, decided in this order:
**intent** (an answerable turn routed to another department), **retrieval** (no gold passage among the
passages shown to the answer model), **generation** (a gold passage was shown, but the reply does not cite
it or refuses), **knowledge gap** (a near-miss turn, which the documents do not cover, got an answer).
Each round takes the largest category that still has untried patches in `configs/skills.yaml`. A patch
changes configuration only (a parameter, or one rule line added to the answer prompt), so applying and
rolling it back needs no code. The patch is run on the same turns. It is kept only if success on its
target set rises with an exact McNemar p below 0.05 and success on the other set does not fall with p
below 0.05; otherwise the configuration rolls back. A kept patch would then be run once on validation
(300 turns, seed 0).

Train, 200 answerable turns (seed 1) and their 117 near-miss rewrites; DeepSeek V4 Flash answering,
Sonnet verifying. At the start: answerable success (answered, citing gold) 0.435, near-miss success (not
answered) 0.880; failures: retrieval 94, generation 18, knowledge gap 14, intent 1.

| Category | Patch | Target before → after | Only before / only after | p | 95% CI of gain | Guard before → after | p |
|---|---|---|---|---|---|---|---|
| retrieval | show 8 passages, not 5 | 0.435 → 0.405 | 18 / 12 | 0.36 | [−0.085, +0.025] | 0.880 → 0.889 | 1 |
| retrieval | session focus (above; 2 replies, 10 passages, bonus 0.1) | 0.435 → 0.425 | 12 / 10 | 0.83 | [−0.055, +0.035] | 0.880 → 0.906 | 0.38 |
| generation | passages cut at 800 words, not 400 | 0.435 → 0.435 | 0 / 0 | 1 | [0, 0] | 0.880 → 0.872 | 1 |
| generation | 3 revisions, not 2 | 0.435 → 0.435 | 0 / 0 | 1 | [0, 0] | 0.880 → 0.880 | 1 |
| generation | rule: cite the passage that states the answer | 0.435 → **0.335** | 24 / 4 | 0.0002 | [−0.150, −0.050] | 0.880 → 0.932 | 0.11 |
| knowledge gap | rule: neighbouring case only → NOT_FOUND | 0.880 → **0.949** | 0 / 8 | 0.008 | [+0.026, +0.120] | 0.435 → **0.330** | 0.0001 |
| knowledge gap | question check (second verifier call) | 0.880 → 0.889 | 0 / 1 | 1 | [0, +0.026] | 0.435 → 0.430 | 1 |
| intent | route less often (thresholds 0.97 / 0.9) | 0.435 → 0.430 | 1 / 0 | 1 | [−0.015, 0] | 0.880 → 0.872 | 1 |
| intent | no routing | 0.435 → 0.405 | 6 / 0 | 0.03 | [−0.055, −0.010] | 0.880 → 0.889 | 1 |

No patch passed, so the configuration stays as it was and there is nothing to confirm on validation. The
rejections are of three kinds:

- **Moves failures between categories.** Both retrieval patches do what they are for: showing 8
  passages cuts retrieval failures from 94 to 81, session focus to 82 (gold in the top 5 on 117 turns
  instead of 105). Generation failures rise in step, from 18 to 37 and to 32: the model does not cite
  the gold it is now shown, and the changed passages cost it turns it got right.
- **Trades one set against the other.** The neighbouring-case rule refuses 8 more near-miss turns and
  breaks none, a significant gain on its target set; but on the answerable set it loses 25 turns and gains
  4 (refusals of answerable turns 49 → 87 of 200). A loop that only looked at the target category would
  have kept it. The citation rule, aimed at generation, does the same (refusals 49 → 79): told to cite
  only a passage that states the answer, the model more often finds none and refuses.
- **Inert here.** Only 0.07% of MultiDoc2Dial passages are longer than 400 words, and 1 of 200 turns uses
  both revisions, so the two parameter patches change no outcome. Turning routing off loses 6 turns: the
  router's single intent failure is not worth the cost of searching every department.

Retrieval is the largest category, but fixing retrieval alone does not raise success: the next work is in
how the answer model uses a gold passage it is shown, judged end to end, not by retrieval recall. About
3.8 USD.

### Replaying a run from its trace

`scripts/replay_trace.py <trace> <run_id>` rebuilds the turn from the trace's first record, runs it
again without budgets (LLM replies come from the response cache, so it costs nothing when no prompt
changed), and compares every node's outputs with the recorded ones: a node marked DIFFERENT is where
code, index or prompt changed since the run; a node that timed out shows what it would have returned.

Reproduce (generation needs VPQA_API_BASE / VPQA_API_KEY / VPQA_MODEL, and VPQA_JUDGE_MODEL for a
separate judge; about 11 USD with Sonnet answering and judging):

```
python scripts/eval_answers.py --dataset md2d --n 300
python scripts/eval_answers.py --dataset md2d --n 300 --withhold gold-doc
python scripts/judge_withheld.py runs/answers/md2d_validation_n300_s0_withheld.jsonl
python scripts/refusal_report.py runs/answers/md2d_validation_n300_s0.jsonl \
    runs/answers/md2d_validation_n300_s0_withheld.jsonl \
    --covered runs/answers/md2d_validation_n300_s0_withheld.covered.json
python scripts/select_refusal_gate.py --train-n 1000
python scripts/make_near_miss.py --split validation --n 300 --seed 0   # the files in data/benchmark are these
python scripts/eval_answers.py --dataset md2d --n 300 --withhold near-miss
python scripts/eval_answers.py --dataset ecfr --split dev --n 100 --versions filter   # or unaware / inject
python scripts/eval_react.py --n 100                        # and --withhold gold-doc, --dataset ecfr
python scripts/eval_fanout.py --k 5 10
python scripts/eval_budgets.py --n 300                      # and --device cpu --scale none 1 0.5
python scripts/eval_memory.py --split train --dialogues 200 --seed 0   # pick the session settings
python scripts/eval_memory.py --split validation --rewrite off short --replies 2 --k 0 10 --bonus 0.0 0.1
python scripts/eval_answer_memory.py --split dev
python scripts/loop.py                                      # train n200 s1; confirms on validation if a patch passes
```

The cheaper answer model: `VPQA_MODEL=deepseek/deepseek-v4-flash VPQA_JUDGE_MODEL=anthropic/claude-sonnet-5.5`,
the same commands with `--out runs/answers_dsv4flash` (and `--answers runs/answers_dsv4flash --out
runs/react_dsv4flash` for eval_react.py). The question check: the three MultiDoc2Dial runs again with
`--check-question` (written next to them with a `_cq` suffix), then
`python scripts/ab_verifier.py runs/answers_dsv4flash`. `--workers N` runs N turns at once, without stage
budgets or latency.

## Layout

```
src/vpqa/
  data/        dataset loaders and normalization to a common Turn/Passage schema
  retrieval/   bm25, dense (bge-m3), RRF fusion, cross-encoder rerank
  pipeline/    the fixed DAG and the verifier
  eval/        metrics and the eval runner
  memory.py    session focus and the version-bound answer memory
  loop.py      failure attribution and paired tests for the offline loop
scripts/       data download, index build, eval entry points
configs/       run configs; skills.yaml, the loop's patches by failure category
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
OpenAI-compatible API set by `VPQA_API_BASE` / `VPQA_API_KEY` / `VPQA_MODEL` (the answer model) and
`VPQA_JUDGE_MODEL` (verifier and eval judges; defaults to `VPQA_MODEL`).
