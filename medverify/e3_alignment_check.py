"""E3 alignment diagnostic (validation only).

    python -m medverify.e3_alignment_check

Do alignment scores separate retrieved GOLD abstracts from other retrieved
abstracts? Reported as mean coverage per group and a simple AUC per feature.
Gold PMIDs are used for this check only, never as a model input.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .context import alignment, parse_claim
from .e2_pipeline import E2_DIR, OUT_DIR
from .retrieval import _read


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = pd.Series(allv).rank().to_numpy()
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def features(split: str = "validation") -> pd.DataFrame:
    run = sorted(E2_DIR.glob(f"scores_*_{split}.csv"))[0]
    df = pd.read_csv(run, dtype={"pmid": str})
    abstract = {d["pmid"]: d["abstract"] for d in _read("corpus.jsonl")}
    gold = {(q["split"], q["claim"]): set(q["gold_pmids"])
            for q in _read("queries.jsonl")}
    rows, bad = [], 0
    for r in df.itertuples():
        p = parse_claim(r.claim)
        if p is None:
            bad += 1
            continue
        a = alignment(p, abstract[r.pmid])
        rows.append({"claim": r.claim, "rank": r.rank, "pmid": r.pmid,
                     "is_gold": r.pmid in gold[(split, r.claim)],
                     "has_gold": bool(gold[(split, r.claim)]), **a})
    out = pd.DataFrame(rows)
    out.attrs["unparsed"] = bad
    return out


def main():
    f = features("validation")
    f = f[f.has_gold]  # only claims that have a gold abstract
    res = {"pairs": int(len(f)), "gold_pairs": int(f.is_gold.sum()),
           "unparsed": f.attrs["unparsed"], "features": {}}
    lines = [f"validation pairs (claims with gold): {len(f)}; "
             f"gold pairs: {int(f.is_gold.sum())}", "",
             "| feature | mean gold | mean non-gold | AUC |",
             "|---|---|---|---|"]
    for c in ("int_cov", "cmp_cov", "out_cov"):
        g = f[f.is_gold][c].dropna().to_numpy()
        n = f[~f.is_gold][c].dropna().to_numpy()
        d = {"mean_gold": float(g.mean()) if len(g) else None,
             "mean_nongold": float(n.mean()) if len(n) else None,
             "auc": auc(g, n)}
        res["features"][c] = d
        lines.append(f"| {c} | {d['mean_gold']:.3f} | "
                     f"{d['mean_nongold']:.3f} | {d['auc']:.3f} |")
    f["align_mean"] = f[["int_cov", "cmp_cov", "out_cov"]].mean(axis=1)
    g = f[f.is_gold].align_mean.to_numpy()
    n = f[~f.is_gold].align_mean.to_numpy()
    res["align_mean_auc"] = auc(g, n)
    lines.append(f"| mean of three | {g.mean():.3f} | {n.mean():.3f} | "
                 f"{res['align_mean_auc']:.3f} |")
    # rank-only baseline: does rank already separate gold?
    res["rank_auc_lower_rank_is_gold"] = auc(
        -f[f.is_gold]["rank"].to_numpy().astype(float),
        -f[~f.is_gold]["rank"].to_numpy().astype(float))
    lines.append(f"\nrank alone (rank 1 = better) AUC: "
                 f"{res['rank_auc_lower_rank_is_gold']:.3f}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "e3_alignment_check.json").write_text(json.dumps(res, indent=2))
    (OUT_DIR / "e3_alignment_check.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
