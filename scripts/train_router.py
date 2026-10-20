"""Train the department router on the train split and pick its routing text and confidence threshold.

Train dialogues are split 80/20. The router is fit on the 80%; on the held-out 20% the routing text is
picked by accuracy, then the confidence threshold by RRF Recall@5 (gated query) of routed retrieval,
separately for first turns (one utterance to route on) and follow-ups. A threshold above 1 means never
route. Each held-out turn is retrieved twice, unfiltered and restricted to the predicted domain, so the
sweep needs no extra retrieval. The final router is refit on all of train and saved with its settings.

    python scripts/train_router.py
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
from pathlib import Path

import numpy as np
import yaml

from vpqa.data.multidoc2dial import load_passages, load_turns
from vpqa.eval.metrics import recall_at_k
from vpqa.retrieval.bm25 import BM25Index
from vpqa.retrieval.dense import DenseIndex
from vpqa.retrieval.fusion import rrf
from vpqa.retrieval.query import build_query
from vpqa.retrieval.route import DomainRouter, routing_text

RAW = Path("data/raw/multidoc2dial")
MODES = ("current", "dialogue")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--sample", type=int, default=4000, help="held-out answerable turns for the threshold sweep")
    ap.add_argument("--thresholds", type=float, nargs="+",
                    default=[0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.01])
    ap.add_argument("--out", default="runs/router")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    rcfg, qcfg = cfg["retrieval"], cfg["query"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    doc_path = RAW / "multidoc2dial_doc.json"
    passages = load_passages(doc_path)
    turns = load_turns(RAW / "multidoc2dial_dial_train.json", doc_path)
    dialogs = sorted({t.dialog_id for t in turns})
    held = set(random.Random(0).sample(dialogs, len(dialogs) // 5))
    is_dev = np.array([t.dialog_id in held for t in turns])
    labels = np.array([t.domain for t in turns])
    first = np.array([not t.history for t in turns])
    print(f"{len(turns)} train turns, {is_dev.sum()} held out ({len(held)} dialogues)")

    name = rcfg["dense_model"]
    dense = DenseIndex(passages, name, cache=Path("data/index") / f"{name.replace('/', '__')}.npy")
    emb = {}
    for mode in MODES:
        cache = out / f"train_{mode}.npy"
        if cache.exists():
            emb[mode] = np.load(cache)
        else:
            emb[mode] = dense.encode([routing_text(t, mode) for t in turns])
            np.save(cache, emb[mode])

    report = {"accuracy": {}}
    for mode in MODES:
        router = DomainRouter().fit(emb[mode][~is_dev], list(labels[~is_dev]))
        pred, _ = router.predict(emb[mode][is_dev])
        hit = np.array(pred) == labels[is_dev]
        report["accuracy"][mode] = {"all": hit.mean(), "first_turn": hit[first[is_dev]].mean(),
                                    "follow_up": hit[~first[is_dev]].mean()}
        print(mode, {k: round(v, 3) for k, v in report["accuracy"][mode].items()})
    mode = max(MODES, key=lambda m: report["accuracy"][m]["all"])

    # threshold: routed vs unrouted RRF on held-out answerable turns
    router = DomainRouter().fit(emb[mode][~is_dev], list(labels[~is_dev]))
    dev_idx = [i for i in np.flatnonzero(is_dev) if turns[i].answerable]
    dev_idx = sorted(random.Random(0).sample(dev_idx, min(args.sample, len(dev_idx))))
    dev = [turns[i] for i in dev_idx]
    pred, conf = router.predict(emb[mode][dev_idx])
    queries = [build_query(t, "gated", qcfg["history_turns"], qcfg["max_words"], qcfg["short_words"]) for t in dev]
    bm25 = BM25Index(passages)

    def retrieve(domains):
        d_runs = dense.search_many(queries, rcfg["dense_k"], domains)
        return [rrf([bm25.search(q, rcfg["bm25_k"], d), dr], k=rcfg["rrf_k"])
                for q, d, dr in zip(queries, domains, d_runs)]

    plain, routed, oracle = retrieve([None] * len(dev)), retrieve(pred), retrieve([t.domain for t in dev])

    def r5(runs):
        return float(np.mean([recall_at_k(r, t.gold_pids, 5) for r, t in zip(runs, dev)]))

    report.update(mode=mode, n_threshold_turns=len(dev), no_routing_R5=r5(plain), oracle_R5=r5(oracle))
    print(f"routing text: {mode}; no routing R@5 {r5(plain):.3f}, oracle {r5(oracle):.3f}")
    threshold = {}
    for group, keep in [("first_turn", lambda t: not t.history), ("follow_up", lambda t: bool(t.history))]:
        idx = [i for i, t in enumerate(dev) if keep(t)]

        def r5_at(th):  # R@5 over this group's turns when routing at confidence >= th
            return float(np.mean([recall_at_k(routed[i] if conf[i] >= th else plain[i], dev[i].gold_pids, 5)
                                  for i in idx]))

        sweep = {f"{th:g}": r5_at(th) for th in args.thresholds}
        threshold[group] = float(max(sweep, key=lambda k: (sweep[k], float(k))))
        report[f"sweep_R5_{group}"] = sweep
        print(f"{group} ({len(idx)} turns)")
        for th, v in sweep.items():
            print(f"  threshold {th}: R@5 {v:.3f}" + ("  <- picked" if float(th) == threshold[group] else ""))
    report["threshold"] = threshold

    final = DomainRouter().fit(emb[mode], list(labels))
    with open(out / "router.pkl", "wb") as f:
        pickle.dump({"router": final, "mode": mode, "threshold": threshold}, f)
    (out / "report.json").write_text(json.dumps(report, indent=2, default=float))


if __name__ == "__main__":
    main()
