"""Step 7, E2: hybrid retrieval (top-3) + E1 BioBERT verifier + aggregation.

    python -m medverify.e2_pipeline build                      # local
    python -m medverify.e2_pipeline infer --runs RUN_DIR ...   # Colab (torch)
    python -m medverify.e2_pipeline evaluate                   # local

build   : top-k hybrid retrieval per claim (dense weight frozen from the
          validation-tuned value in results/retrieval/retrieval.json).
          Writes data/e2/retrieved_{validation,test}.csv.
infer   : scores each (claim, retrieved abstract) pair with an E1 checkpoint
          (head_tail truncation) -> data/e2/scores_{run}_{split}.csv.
evaluate: aggregates per claim and reports macro F1, accuracy, confusion
          matrix, seen/unseen, mean +- sd over runs.

Claim-level gold: the non-NEI label of the claim's paired papers (most
common if several); NEI if all its paired papers are NEI.
Aggregation: among retrieved abstracts whose argmax is non-NEI, take the one
with the highest confidence; if none, predict NEI.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .config import LABELS
from .data import load_all, to_pairs
from .retrieval import CORPUS_DIR, ROOT, _ranks, _read, _tok, rrf

E2_DIR = ROOT / "data" / "e2"
OUT_DIR = ROOT / "results" / "e2"
SPLITS = ("validation", "test")
NEI = next(k for k, v in LABELS.items() if v == "NEI")
TOP_K = 3


def build(top_k: int = TOP_K):
    from rank_bm25 import BM25Okapi

    w = json.loads((ROOT / "results/retrieval/retrieval.json")
                   .read_text())["dense_weight"]
    docs, queries = _read("corpus.jsonl"), _read("queries.jsonl")
    pmids = [d["pmid"] for d in docs]
    bm25 = BM25Okapi([_tok(d["abstract"]) for d in docs])
    s_bm = np.array([bm25.get_scores(_tok(q["claim"])) for q in queries])
    s_de = (np.load(CORPUS_DIR / "emb_queries.npy")
            @ np.load(CORPUS_DIR / "emb_docs.npy").T)
    s_hy = rrf([_ranks(s_bm), _ranks(s_de)], [1 - w, w])

    data = load_all()
    gold, conflicts = {}, {s: 0 for s in SPLITS}
    for s in SPLITS:
        p = to_pairs(data[s])
        for claim, g in p.groupby("claim"):
            non = [int(x) for x in g["label"] if x != NEI]
            if not non:
                gold[(s, claim)] = NEI
            else:
                c = Counter(non).most_common()
                conflicts[s] += len(c) > 1
                gold[(s, claim)] = c[0][0]

    E2_DIR.mkdir(parents=True, exist_ok=True)
    for s in SPLITS:
        rows = []
        for qi, q in enumerate(queries):
            if q["split"] != s:
                continue
            for r, j in enumerate(np.argsort(-s_hy[qi])[:top_k], 1):
                rows.append({"claim": q["claim"], "nctId": q["nctId"],
                             "claim_unseen": q["claim_unseen"], "rank": r,
                             "pmid": pmids[j],
                             "gold_claim_label": gold[(s, q["claim"])]})
        df = pd.DataFrame(rows)
        df.to_csv(E2_DIR / f"retrieved_{s}.csv", index=False)
        print(s, "claims", df["claim"].nunique(), "pairs", len(df),
              "tie-conflicts", conflicts[s])
    print("dense weight", w, "top_k", top_k)


def infer(runs: list[str], policy: str = "head_tail", batch: int = 32):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from .config import BIOBERT_CHECKPOINT, MAX_LENGTH
    from .encoding import encode_pairs

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    abstract = {d["pmid"]: d["abstract"] for d in _read("corpus.jsonl")}
    for run in runs:
        run = Path(run)
        ckpt = run
        if not (run / "config.json").exists():
            best = None
            m = run / "metrics.json"
            if m.exists():
                best = json.loads(m.read_text()).get("best_checkpoint")
            cands = sorted((run / "checkpoints").glob("checkpoint-*"))
            ckpt = (Path(best) if best and Path(best).exists()
                    else (cands[-1] if cands else None))
            if ckpt is None:
                raise SystemExit(f"no checkpoint under {run}")
        try:
            tok = AutoTokenizer.from_pretrained(ckpt)
        except Exception:
            tok = AutoTokenizer.from_pretrained(BIOBERT_CHECKPOINT)
        model = AutoModelForSequenceClassification.from_pretrained(ckpt)
        model.to(dev).eval()
        for s in SPLITS:
            df = pd.read_csv(E2_DIR / f"retrieved_{s}.csv",
                             dtype={"pmid": str})
            enc = encode_pairs(tok, df["claim"].tolist(),
                               [abstract[p] for p in df["pmid"]],
                               policy, MAX_LENGTH)
            probs = []
            with torch.no_grad():
                for i in range(0, len(df), batch):
                    feats = [{k: enc[k][j] for k in enc}
                             for j in range(i, min(i + batch, len(df)))]
                    b = tok.pad(feats, return_tensors="pt").to(dev)
                    probs.append(torch.softmax(model(**b).logits, -1).cpu())
            probs = torch.cat(probs).numpy()
            for i, n in LABELS.items():
                df[f"prob_{n}"] = probs[:, i]
            out = E2_DIR / f"scores_{run.name}_{s}.csv"
            df.to_csv(out, index=False)
            print("wrote", out)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    cols = [f"prob_{LABELS[i]}" for i in sorted(LABELS)]
    for claim, g in df.groupby("claim", sort=False):
        p = g[cols].to_numpy()
        arg = p.argmax(1)
        conf = p.max(1)
        non = [i for i in range(len(g)) if arg[i] != NEI]
        pred = int(arg[max(non, key=lambda i: conf[i])]) if non else NEI
        rows.append({"claim": claim, "pred": pred,
                     "gold": int(g["gold_claim_label"].iloc[0]),
                     "claim_unseen": bool(g["claim_unseen"].iloc[0])})
    return pd.DataFrame(rows)


def evaluate():
    from .metrics import classification_metrics

    files = sorted(E2_DIR.glob("scores_*_validation.csv"))
    if not files:
        raise SystemExit("no scores files; run infer first")
    runs = [f.name[len("scores_"):-len("_validation.csv")] for f in files]
    res = {"runs": runs, "per_run": {}, "baseline_always_nei": {}}
    for run in runs:
        res["per_run"][run] = {}
        for s in SPLITS:
            a = aggregate(pd.read_csv(E2_DIR / f"scores_{run}_{s}.csv"))
            res["per_run"][run][s] = {
                "all": classification_metrics(a["gold"], a["pred"]),
                "unseen": classification_metrics(
                    a[a.claim_unseen]["gold"], a[a.claim_unseen]["pred"])}
    for s in SPLITS:
        a = aggregate(pd.read_csv(E2_DIR / f"scores_{runs[0]}_{s}.csv"))
        res["baseline_always_nei"][s] = classification_metrics(
            a["gold"], [NEI] * len(a))

    lines = ["| run | split | subset | n | macro F1 | acc |",
             "|---|---|---|---|---|---|"]
    summ = {}
    for s in SPLITS:
        for sub in ("all", "unseen"):
            f1 = [res["per_run"][r][s][sub]["macro_f1"] for r in runs]
            ac = [res["per_run"][r][s][sub]["accuracy"] for r in runs]
            n = res["per_run"][runs[0]][s][sub]["n"]
            for r in runs:
                m = res["per_run"][r][s][sub]
                lines.append(f"| {r} | {s} | {sub} | {n} | "
                             f"{m['macro_f1']:.3f} | {m['accuracy']:.3f} |")
            summ[f"{s}_{sub}"] = {
                "macro_f1_mean": float(np.mean(f1)),
                "macro_f1_sd": float(np.std(f1, ddof=1)) if len(f1) > 1 else 0,
                "acc_mean": float(np.mean(ac))}
    res["summary"] = summ
    lines += ["", "Mean +- sd over runs (macro F1):"]
    for k, v in summ.items():
        lines.append(f"- {k}: {v['macro_f1_mean']:.3f} +- "
                     f"{v['macro_f1_sd']:.3f} (acc {v['acc_mean']:.3f})")
    lines += ["", "Always-NEI baseline:"]
    for s in SPLITS:
        b = res["baseline_always_nei"][s]
        lines.append(f"- {s}: macro F1 {b['macro_f1']:.3f}, "
                     f"acc {b['accuracy']:.3f}")
    first = res["per_run"][runs[0]]["test"]["all"]
    lines += ["", f"Test confusion matrix ({runs[0]}; rows=true, cols=pred, "
              f"order {[LABELS[i] for i in sorted(LABELS)]}):",
              str(first["confusion_matrix"])]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "e2_results.json").write_text(json.dumps(res, indent=2))
    (OUT_DIR / "e2_results.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "infer", "evaluate"])
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--top_k", type=int, default=TOP_K)
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.top_k)
    elif a.cmd == "infer":
        infer(a.runs)
    else:
        evaluate()
