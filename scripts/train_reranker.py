"""Fine-tune the cross-encoder reranker on MultiDoc2Dial train with hard negatives.

    python scripts/train_reranker.py --turns 6000 --out models/reranker-md2d
    python scripts/eval_retrieval.py --query-mode gated --rerank 40 --route router --rerank-model models/reranker-md2d

The recipe is the usual one for BGE / DPR-style rerankers: each training group is the gated query of one
turn, one gold passage and `group - 1` hard negatives drawn from the top of the same RRF candidate list
the reranker sees at serving time (BM25 + bge-m3, unrouted, gold passages removed); the loss is softmax
cross-entropy over the group with the gold passage as the target. One epoch, linear warmup and decay.

Dialogues of the train sample the offline loop scores on (`--exclude`) are left out, so that sample stays
unseen. `--dev` turns from further held-out train dialogues give R@1 / R@5 of the reranked candidates
before and after training.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from vpqa.data.multidoc2dial import load_passages, load_turns
from vpqa.retrieval.bm25 import BM25Index
from vpqa.retrieval.dense import DenseIndex
from vpqa.retrieval.fusion import rrf
from vpqa.retrieval.query import build_query

RAW = Path("data/raw/multidoc2dial")


def candidates(turns, queries, bm25, dense, cfg) -> list[list[str]]:
    lists = dense.search_many(queries, cfg["dense_k"], [None] * len(queries))
    return [rrf([bm25.search(q, cfg["bm25_k"]), d], k=cfg["rrf_k"]) for q, d in zip(queries, lists)]


@torch.no_grad()
def evaluate(model, tok, text, queries, cands, turns, device, max_length) -> dict:
    model.eval()
    r1 = r5 = 0
    for q, c, t in zip(queries, cands, turns):
        enc = tok([q] * len(c), [text[p] for p in c], truncation="only_second", max_length=max_length,
                  padding=True, return_tensors="pt").to(device)
        s = model(**enc).logits.squeeze(-1).float().tolist()
        ranked = [p for _, p in sorted(zip(s, c), key=lambda x: -x[0])]
        r1 += bool(set(ranked[:1]) & set(t.gold_pids))
        r5 += len(set(ranked[:5]) & set(t.gold_pids)) / len(t.gold_pids)
    model.train()
    return {"R@1": r1 / len(turns), "R@5": r5 / len(turns), "n": len(turns)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--model", default="BAAI/bge-reranker-v2-m3")
    ap.add_argument("--turns", type=int, default=6000, help="training turns (groups), sampled from train")
    ap.add_argument("--dev", type=int, default=400)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--neg-depth", type=int, default=30, help="hard negatives come from this top of the RRF list")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup", type=float, default=0.1)
    ap.add_argument("--accum", type=int, default=2, help="groups per optimizer step")
    ap.add_argument("--max-length", type=int, default=384)
    ap.add_argument("--exclude", type=Path, default=Path("runs/answers_dsv4flash/md2d_train_n200_s1.jsonl"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("models/reranker-md2d"))
    ap.add_argument("--save-every", type=int, default=500, help="groups between checkpoints; a rerun resumes from the last")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    qcfg = cfg["query"]
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    doc_path = RAW / "multidoc2dial_doc.json"
    passages = load_passages(doc_path)
    text = {p.pid: p.text for p in passages}
    turns = [t for t in load_turns(RAW / "multidoc2dial_dial_train.json", doc_path) if t.answerable]
    excluded = {json.loads(l)["id"].split("/")[0] for l in args.exclude.read_text().splitlines()}
    dialogs = sorted({t.dialog_id for t in turns} - excluded)
    rng.shuffle(dialogs)
    dev_dialogs = set(dialogs[: len(dialogs) // 20])
    dev = [t for t in turns if t.dialog_id in dev_dialogs]
    train = [t for t in turns if t.dialog_id not in dev_dialogs and t.dialog_id not in excluded]
    dev, train = rng.sample(dev, min(args.dev, len(dev))), rng.sample(train, min(args.turns, len(train)))
    print(f"{len(train)} train turns, {len(dev)} dev turns, {len(excluded)} dialogues excluded")

    q = lambda t: build_query(t, "gated", qcfg["history_turns"], qcfg["max_words"], qcfg["short_words"])
    bm25 = BM25Index(passages)
    dense = DenseIndex(passages, cfg["retrieval"]["dense_model"], cache=Path("data/index/BAAI__bge-m3.npy"))
    train_q, dev_q = [q(t) for t in train], [q(t) for t in dev]
    train_c, dev_c = candidates(train, train_q, bm25, dense, cfg["retrieval"]), candidates(dev, dev_q, bm25, dense, cfg["retrieval"])
    del dense

    groups = []
    for t, query, c in zip(train, train_q, train_c):
        negs = [p for p in c[: args.neg_depth] if p not in t.gold_pids]
        if len(negs) < args.group - 1:
            negs += rng.sample([p for p in text if p not in t.gold_pids and p not in negs], args.group - 1 - len(negs))
        groups.append((query, [rng.choice(t.gold_pids)] + rng.sample(negs, args.group - 1)))
    print(f"gold in candidates: {sum(bool(set(c) & set(t.gold_pids)) for t, c in zip(train, train_c)) / len(train):.3f}")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(device)
    model.gradient_checkpointing_enable()
    steps = len(groups) // args.accum
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    warm = max(1, int(args.warmup * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, max(0.0, (steps - s) / (steps - warm))))
    ckpt, start = args.out / "checkpoint.pt", 0
    if ckpt.exists():  # the groups come from the seed, so the run picks up where it stopped
        state = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        start, log = state["done"], state["log"]
        print(f"resumed after {start} groups")
    else:
        log = {"args": {k: str(v) for k, v in vars(args).items()}}
        log["dev_before"] = evaluate(model, tok, text, dev_q, dev_c, dev, device, 512)
        print("dev before:", log["dev_before"])

    target = torch.zeros(1, dtype=torch.long, device=device)
    t0, running = time.time(), []
    model.train()
    for i, (query, pids) in enumerate(groups):
        if i < start:
            continue
        enc = tok([query] * len(pids), [text[p] for p in pids], truncation="only_second", max_length=args.max_length,
                  padding=True, return_tensors="pt").to(device)
        loss = torch.nn.functional.cross_entropy(model(**enc).logits.view(1, -1), target) / args.accum
        loss.backward()
        running.append(loss.item() * args.accum)
        if (i + 1) % args.accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
        if (i + 1) % 200 == 0:
            el, done = time.time() - t0, i + 1 - start
            print(f"{i + 1}/{len(groups)} loss {sum(running) / len(running):.3f} "
                  f"{el / done:.2f} s/group, eta {el / done * (len(groups) - i - 1) / 60:.0f} min", flush=True)
            running = []
        if (i + 1) % args.save_every == 0 and (i + 1) % args.accum == 0 and i + 1 < len(groups):
            args.out.mkdir(parents=True, exist_ok=True)
            log["train_minutes"] = log.get("train_minutes", 0) + (time.time() - t0) / 60
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "done": i + 1, "log": log}, ckpt.with_suffix(".tmp"))
            ckpt.with_suffix(".tmp").replace(ckpt)
            log["train_minutes"] -= (time.time() - t0) / 60  # added back in full at the end
    log["train_minutes"] = log.get("train_minutes", 0) + (time.time() - t0) / 60
    log["dev_after"] = evaluate(model, tok, text, dev_q, dev_c, dev, device, 512)
    print("dev after:", log["dev_after"])

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    (args.out / "train_log.json").write_text(json.dumps(log, indent=2))
    ckpt.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
