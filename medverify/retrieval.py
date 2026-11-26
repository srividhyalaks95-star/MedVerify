"""Step 5/6: BM25 + MedCPT retrieval, RRF fusion, evaluation.

Run from Final_Project/:
    python -m medverify.retrieval embed      # needs torch (use Colab GPU)
    python -m medverify.retrieval evaluate   # numpy + rank_bm25 only

embed writes data/corpus/emb_{docs,queries}.npy (row order = corpus.jsonl /
queries.jsonl). The fusion weight is tuned on validation only, then frozen
for test. Metrics use queries with gold PMIDs only.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "data" / "corpus"
OUT_DIR = ROOT / "results" / "retrieval"
ART = "ncbi/MedCPT-Article-Encoder"
QRY = "ncbi/MedCPT-Query-Encoder"
RRF_K = 60
KS = (5, 10)


def _read(name):
    with open(CORPUS_DIR / name) as f:
        return [json.loads(l) for l in f]


# ---------------- embedding (GPU/torch) ----------------
def embed(batch_size: int = 32):
    import torch
    from transformers import AutoModel, AutoTokenizer

    dev = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    docs, queries = _read("corpus.jsonl"), _read("queries.jsonl")

    def run(name, texts, max_len):
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModel.from_pretrained(name).to(dev).eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                enc = tok(texts[i:i + batch_size], truncation=True,
                          padding=True, max_length=max_len,
                          return_tensors="pt").to(dev)
                out.append(model(**enc).last_hidden_state[:, 0, :].cpu())
        return torch.cat(out).numpy().astype("float32")

    np.save(CORPUS_DIR / "emb_docs.npy",
            run(ART, [d["abstract"] for d in docs], 512))
    np.save(CORPUS_DIR / "emb_queries.npy",
            run(QRY, [q["claim"] for q in queries], 64))
    print("saved embeddings on", dev)


# ---------------- ranking ----------------
def _tok(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def _ranks(scores):
    """scores: (nq, nd) -> rank matrix (1 = best)."""
    order = np.argsort(-scores, axis=1)
    ranks = np.empty_like(order)
    rows = np.arange(scores.shape[0])[:, None]
    ranks[rows, order] = np.arange(1, scores.shape[1] + 1)[None, :]
    return ranks


def rrf(rank_lists, weights):
    return sum(w / (RRF_K + r) for w, r in zip(weights, rank_lists))


def _metrics(scores, idx, gold_sets, pmids):
    out = {"n": len(idx)}
    if not idx:
        return out
    order = np.argsort(-scores[idx], axis=1)
    rec = {k: [] for k in KS}
    rr = []
    for row, qi in zip(order, idx):
        ranked = [pmids[j] for j in row]
        g = gold_sets[qi]
        for k in KS:
            rec[k].append(len(g & set(ranked[:k])) / len(g))
        pos = next((i + 1 for i, p in enumerate(ranked) if p in g), None)
        rr.append(1.0 / pos if pos else 0.0)
    for k in KS:
        out[f"recall@{k}"] = round(float(np.mean(rec[k])), 4)
    out["mrr"] = round(float(np.mean(rr)), 4)
    return out


def evaluate():
    from rank_bm25 import BM25Okapi

    docs, queries = _read("corpus.jsonl"), _read("queries.jsonl")
    pmids = [d["pmid"] for d in docs]
    bm25 = BM25Okapi([_tok(d["abstract"]) for d in docs])
    s_bm = np.array([bm25.get_scores(_tok(q["claim"])) for q in queries])
    ed = np.load(CORPUS_DIR / "emb_docs.npy")
    eq = np.load(CORPUS_DIR / "emb_queries.npy")
    s_de = eq @ ed.T  # MedCPT uses dot product
    r_bm, r_de = _ranks(s_bm), _ranks(s_de)
    gold = [set(q["gold_pmids"]) for q in queries]

    def idx(split, unseen=None):
        return [i for i, q in enumerate(queries)
                if q["split"] == split and gold[i]
                and (unseen is None or q["claim_unseen"] == unseen)]

    # tune dense weight on validation only (objective: MRR)
    grid = [round(w, 1) for w in np.arange(0, 1.01, 0.1)]
    tuning = {}
    for w in grid:
        sc = rrf([r_bm, r_de], [1 - w, w])
        tuning[w] = _metrics(sc, idx("validation"), gold, pmids)["mrr"]
    best_w = max(grid, key=lambda w: (tuning[w], -abs(w - 0.5)))
    s_hy = rrf([r_bm, r_de], [1 - best_w, best_w])

    systems = {"bm25": s_bm, "medcpt": s_de, "hybrid_rrf": s_hy}
    res = {"dense_weight": best_w, "val_tuning_mrr": tuning, "results": {}}
    for name, sc in systems.items():
        res["results"][name] = {
            sp: {"all": _metrics(sc, idx(sp), gold, pmids),
                 "unseen": _metrics(sc, idx(sp, True), gold, pmids)}
            for sp in ("validation", "test")}

    # failure inspection: test misses at top-10 for hybrid
    misses = []
    for qi in idx("test"):
        top = [pmids[j] for j in np.argsort(-s_hy[qi])[:10]]
        if not gold[qi] & set(top):
            misses.append({"claim": queries[qi]["claim"],
                           "gold": sorted(gold[qi]), "top3": top[:3]})
    res["test_hybrid_misses_top10"] = len(misses)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "retrieval.json").write_text(json.dumps(res, indent=2))
    (OUT_DIR / "retrieval_misses.json").write_text(
        json.dumps(misses[:30], indent=2))

    lines = [f"Dense weight (val-tuned): {best_w}", "",
             "| system | split | subset | n | R@5 | R@10 | MRR |",
             "|---|---|---|---|---|---|---|"]
    for name, d in res["results"].items():
        for sp, sub in d.items():
            for k, m in sub.items():
                lines.append(f"| {name} | {sp} | {k} | {m['n']} | "
                             f"{m.get('recall@5')} | {m.get('recall@10')} | "
                             f"{m.get('mrr')} |")
    (OUT_DIR / "retrieval.md").write_text("\n".join(lines))
    print("\n".join(lines))
    print("test hybrid misses@10:", len(misses))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "evaluate"
    {"embed": embed, "evaluate": evaluate}[cmd]()
