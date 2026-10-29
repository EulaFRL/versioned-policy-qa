# Evaluation summary

Results below summarize locally recorded runs; this repository excludes run logs, model weights, raw MultiDoc2Dial data,
near-miss JSONL files, and the eCFR candidate-pair audit log. It includes only the derived eCFR question set, which contains citations rather than regulation text. The README describes the methods and commands. Dataset-derived numbers depend
on the sampled data and models noted here.

## Fine-tuned reranker

On 4,201 MultiDoc2Dial validation turns, fine-tuning raised rerank@40 Hit@1 from 0.225 to 0.362 and Hit@5
from 0.550 to 0.605. In a 300-turn end-to-end validation run, answerable success rose from 0.390 to 0.463.
Refusal on 167 withheld turns judged uncovered by both top-5 lists fell from 0.611 to 0.539; near-miss
refusal changed from 0.782 to 0.741. This trade-off leaves the reranker experimental and off by default.
These are DeepSeek runs with eight workers.

Three coverage judges disagreed: they marked 109, 91, and 101 of 300 withheld turns as covered. A planned
10-item human check was unfinished, so the coverage-based refusal comparison is uncertain. The latency runs
had GPU contention and retrieval fallbacks; they do not establish a reliable end-to-end speed gain.

## Condition check

On three repeated training samples, the `note` condition-check setting raised near-miss refusal from 0.866
to 0.917, while answerable success changed from 0.430 to 0.418. On three validation samples, near-miss
refusal rose from 0.793 to 0.889 and answerable success changed from 0.384 to 0.377. The check is enabled
in the baseline config. It adds an answer-model call; measured latency was not available.

The loop's two-point guard is not a statistical non-inferiority test: a large loss that is not significant
can still pass. The results are a small-sample engineering evaluation, not a guarantee of production error
rates.
