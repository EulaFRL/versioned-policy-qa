"""Organisational knowledge: reusing verified answers across questions and dates, on the eCFR pairs.

    python scripts/eval_answer_memory.py --split dev

Each eCFR pair asks one question on two dates, and the paragraph it is about reads differently on them
(scripts/make_version_qa.py). The memory is filled by running the pipeline on every pair's question at
its old date and storing the answers that come out verified and cite at least one passage
(vpqa.memory.AnswerMemory). It is then asked,
in other words (a paraphrase by VPQA_MODEL), at two dates:

- same window: a date inside the validity window of the old gold version, other than the fill date.
  Reusing the stored answer is right here: the text it cites is still the law;
- new version: the pair's new date, when the old gold version is no longer in force. Reusing the stored
  answer cites a version that no longer applies.

Every lookup query also runs through the pipeline, which is what a policy falls back to when it does not
reuse. Policies: `none` (no memory), `unbound` (reuse the most similar answer at or above the threshold)
and `bound` (reuse only an answer whose cited versions are all in force on the query date). Per query
set: share reused, share reused from another pair's question, Cite-Hit (the final citations include the
gold version) and Stale (a final citation is not in force on the date), both over the turns not refused.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import numpy as np
import yaml

from vpqa import llm
from vpqa.data.ecfr import load_versions
from vpqa.memory import AnswerMemory
from vpqa.pipeline.dag import Pipeline
from vpqa.pipeline.verifier import in_force
from vpqa.schema import Turn

PARAPHRASE = """Rewrite this question in other words. Keep its meaning and every condition in it; do not answer it, add details or drop any. Reply with the rewritten question only.

Question: {question}"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--threshold", nargs="+", type=float, default=[0.8, 0.85, 0.9, 0.95])
    ap.add_argument("--n", type=int, default=0, help="first n pairs only (0: all)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--qa", type=Path, default=Path("data/benchmark/ecfr_version_qa.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("runs/answer_memory"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    items = [json.loads(l) for l in args.qa.read_text().splitlines()]
    items = [it for it in items if it["split"] == args.split]
    by_qid = {it["qid"]: it for it in items}
    pairs = [(it, by_qid[o]) for it in items if it["version"] == "old"
             for o in it["other_versions"] if o in by_qid and by_qid[o]["version"] == "new"
             and by_qid[o]["question"] == it["question"]]
    pairs = pairs[:args.n] if args.n else pairs
    pipe = Pipeline(load_versions(), yaml.safe_load(open(args.config)), Path("data/index/BAAI__bge-m3__ecfr.npy"),
                    generate=True, version_filter=True, timeouts=False)

    def window_date(it: dict) -> date:
        """A date in the old gold version's validity window, away from the fill date."""
        p = pipe.by_pid[it["gold_pid"]]
        start, end, fill = p.effective_from, p.effective_to, date.fromisoformat(it["as_of"])
        for f in (0.5, 0.25, 0.75):
            d = start + (end - start) * f
            if d != fill and abs((d - fill).days) > 30:
                return d
        return start if start != fill else end - (end - start) / 2

    def paraphrase(q: str) -> str:
        reply = llm.chat([{"role": "user", "content": PARAPHRASE.format(question=q)}], max_tokens=200)
        return next((l.strip().strip('"') for l in reply.splitlines() if l.strip()), "") or q

    with ThreadPoolExecutor(args.workers) as pool:
        para = list(pool.map(paraphrase, [old["question"] for old, _ in pairs]))
    fill = [Turn(old["qid"], 0, "", [], old["question"], [old["gold_pid"]], as_of=date.fromisoformat(old["as_of"]))
            for old, _ in pairs]
    queries = {"same window": [Turn(old["qid"], 1, "", [], q, [old["gold_pid"]], as_of=window_date(old))
                               for (old, _), q in zip(pairs, para)],
               "new version": [Turn(old["qid"], 2, "", [], q, [new["gold_pid"]], as_of=date.fromisoformat(new["as_of"]))
                               for (old, new), q in zip(pairs, para)]}
    for t in queries["same window"]:
        assert in_force(pipe.by_pid[t.gold_pids[0]], t.as_of)

    def run(t: Turn) -> dict:
        state, _ = pipe.run(t)
        return {k: state[k] for k in ("final", "final_pids", "outcome")}

    with ThreadPoolExecutor(args.workers) as pool:
        filled = list(pool.map(run, fill))
        ran = {k: list(pool.map(run, ts)) for k, ts in queries.items()}
    print(f"{len(pairs)} pairs; pipeline runs done, cost {llm.TOTALS['cost']:.3f}", flush=True)

    encode = pipe.dense.encode
    fill_vec = encode([t.query for t in fill])
    query_vec = encode(para)
    stored = [i for i, f in enumerate(filled) if f["outcome"] == "answered" and f["final_pids"]]
    sims_own = [float(fill_vec[i] @ query_vec[i]) for i in range(len(pairs))]

    rows, results = [], []
    for policy, thresholds in (("none", [None]), ("unbound", args.threshold), ("bound", args.threshold)):
        for tau in thresholds:
            mem = None
            if policy != "none":
                mem = AnswerMemory(encode, pipe.by_pid, threshold=tau, bind=policy == "bound")
                for i in stored:
                    mem.store(fill[i].query, filled[i]["final"], filled[i]["final_pids"], fill_vec[i])
                owner = {id(e): i for e, i in zip(mem.entries, stored)}
            for name, ts in queries.items():
                out = []
                for i, t in enumerate(ts):
                    entry, why = mem.lookup(t.query, t.as_of, query_vec[i]) if mem else (None, "miss")
                    if entry:
                        cited, refused, src = entry.cited, False, owner[id(entry)]
                    else:
                        cited, refused, src = ran[name][i]["final_pids"], ran[name][i]["outcome"] == "refused", None
                    out.append({"pair": i, "reused": entry is not None, "lookup": why, "from_pair": src,
                                "refused": refused, "cited": cited,
                                "hit": bool(set(cited) & set(t.gold_pids)),
                                "stale": any(not in_force(pipe.by_pid[p], t.as_of) for p in cited)})
                given = [o for o in out if not o["refused"]]
                results.append({"policy": policy, "threshold": tau, "set": name, "n": len(out),
                                "reused": np.mean([o["reused"] for o in out]),
                                "reused_other_pair": np.mean([o["reused"] and o["from_pair"] != o["pair"] for o in out]),
                                "blocked_stale": np.mean([o["lookup"] == "stale" for o in out]),
                                "refused": np.mean([o["refused"] for o in out]),
                                "Cite-Hit": np.mean([o["hit"] for o in given]),
                                "Stale": np.mean([o["stale"] for o in given])})
                rows += [{"policy": policy, "threshold": tau, "set": name, **o} for o in out]

    summary = {"pairs": len(pairs), "stored": len(stored),
               "fill": {"Cite-Hit": np.mean([bool(set(f["final_pids"]) & {old["gold_pid"]})
                                             for f, (old, _) in zip(filled, pairs) if f["outcome"] != "refused"]),
                        "answered": len(stored) / len(pairs)},
               "paraphrase_similarity_p10_p50": [float(np.percentile(sims_own, 10)), float(np.percentile(sims_own, 50))],
               "llm": llm.TOTALS, "results": results}
    (args.out / f"ecfr_{args.split}.json").write_text(json.dumps(summary, indent=1, default=float))
    (args.out / f"ecfr_{args.split}.jsonl").write_text("".join(json.dumps(r, default=str) + "\n" for r in rows))
    print(f"{len(stored)} of {len(pairs)} fill answers verified and stored; fill {summary['fill']}; "
          f"own-paraphrase similarity p10/p50 {summary['paraphrase_similarity_p10_p50']}")
    print("\n| Policy | Threshold | Set | Reused | From another pair | Blocked as stale | Refused | Cite-Hit | Stale |")
    print("|---" * 9 + "|")
    for r in results:
        print(f"| {r['policy']} | {r['threshold'] or '-'} | {r['set']} | {r['reused']:.3f} | {r['reused_other_pair']:.3f} | "
              f"{r['blocked_stale']:.3f} | {r['refused']:.3f} | {r['Cite-Hit']:.3f} | {r['Stale']:.3f} |")
    print(f"cost {llm.TOTALS['cost']:.3f}")


if __name__ == "__main__":
    main()
