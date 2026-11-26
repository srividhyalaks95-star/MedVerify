"""Step 4: evidence corpus and retrieval queries.

Run from Final_Project/:
    python -m medverify.corpus

Writes data/corpus/corpus.jsonl, queries.jsonl and corpus_report.json.
The corpus holds only PMID + abstract: no labels, claims, nctId or pValues.
Gold PMIDs live only in queries.jsonl and are used for evaluation, never
for indexing.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import LABELS
from .data import load_all, to_pairs

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "corpus"
NEI_ID = next(k for k, v in LABELS.items() if v == "NEI")


def build() -> dict:
    data = load_all()
    pairs = {s: to_pairs(df) for s, df in data.items()}
    allp = pd.concat([p.assign(split=s) for s, p in pairs.items()],
                     ignore_index=True)
    allp["abstract"] = allp["article_abstract"].fillna("").str.strip()

    # ---- corpus: one record per PMID ----
    n_variants = allp.groupby("PMID")["abstract"].nunique()
    conflicting = n_variants[n_variants > 1].index.tolist()
    nonempty = allp[allp["abstract"] != ""]
    corpus = (nonempty.assign(_len=nonempty["abstract"].str.len())
              .sort_values("_len", ascending=False)
              .drop_duplicates("PMID")[["PMID", "abstract"]]
              .sort_values("PMID").reset_index(drop=True))
    empty_pmids = sorted(set(allp["PMID"]) - set(corpus["PMID"]))

    # ---- queries: unique (split, claim); gold = non-NEI paired PMIDs ----
    allp["is_gold"] = allp["label"] != NEI_ID
    queries = []
    for (split, claim), g in allp.groupby(["split", "claim"], sort=False):
        gold = sorted(set(g.loc[g["is_gold"], "PMID"]) - set(empty_pmids))
        queries.append({
            "split": split,
            "claim": claim,
            "nctId": str(g["nctId"].iloc[0]),
            "claim_unseen": bool(g["claim_unseen"].iloc[0]),
            "n_rows": int(len(g)),
            "gold_pmids": [str(p) for p in gold],
        })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "corpus.jsonl", "w") as f:
        for r in corpus.itertuples():
            f.write(json.dumps({"pmid": str(r.PMID),
                                "abstract": r.abstract}) + "\n")
    with open(OUT_DIR / "queries.jsonl", "w") as f:
        for q in queries:
            f.write(json.dumps(q) + "\n")

    per_split = {}
    for s in ("train", "validation", "test"):
        qs = [q for q in queries if q["split"] == s]
        with_gold = [q for q in qs if q["gold_pmids"]]
        per_split[s] = {
            "queries": len(qs),
            "queries_with_gold": len(with_gold),
            "with_gold_unseen": sum(q["claim_unseen"] for q in with_gold),
        }
    report = {
        "rows": int(len(allp)),
        "corpus_docs": int(len(corpus)),
        "pmids_with_conflicting_abstracts": len(conflicting),
        "pmids_with_empty_abstract": len(empty_pmids),
        "queries": len(queries),
        "per_split": per_split,
        "excluded_from_corpus": ["label", "claim", "nctId", "pValues"],
    }
    (OUT_DIR / "corpus_report.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
