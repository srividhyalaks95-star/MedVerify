"""E2 diagnostic: where do errors come from?

    python -m medverify.e2_diagnostic

1. Retrieval: for claims with gold PMIDs, is a gold abstract in the top 3?
   Accuracy split by hit / miss.
2. NEI claims: how often does any retrieved abstract get a non-NEI vote?
3. Aggregation variants (descriptive; the rule is chosen on validation).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .config import LABELS
from .e2_pipeline import E2_DIR, NEI, OUT_DIR, SPLITS, aggregate
from .metrics import classification_metrics
from .retrieval import _read

COLS = [f"prob_{LABELS[i]}" for i in sorted(LABELS)]


def agg_variants(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {"top3_any_non_nei (variant)": aggregate(df, "top3_any_non_nei")}
    for name in ("mean_prob", "top1_only (primary)"):
        rows = []
        for claim, g in df.groupby("claim", sort=False):
            p = g.sort_values("rank")[COLS].to_numpy()
            pred = (int(p.mean(0).argmax()) if name == "mean_prob"
                    else int(p[0].argmax()))
            rows.append({"claim": claim, "pred": pred,
                         "gold": int(g["gold_claim_label"].iloc[0]),
                         "claim_unseen": bool(g["claim_unseen"].iloc[0])})
        out[name] = pd.DataFrame(rows)
    return out


def main(run: str | None = None):
    files = sorted(E2_DIR.glob("scores_*_validation.csv"))
    run = run or files[0].name[len("scores_"):-len("_validation.csv")]
    gold_pm = {(q["split"], q["claim"]): set(q["gold_pmids"])
               for q in _read("queries.jsonl")}
    res = {"run": run}
    lines = [f"Run: {run}", ""]
    for s in SPLITS:
        df = pd.read_csv(E2_DIR / f"scores_{run}_{s}.csv",
                         dtype={"pmid": str})
        agg = aggregate(df, "top3_any_non_nei")
        hits = []
        for claim, g in df.groupby("claim", sort=False):
            gp = gold_pm[(s, claim)]
            hits.append(None if not gp else bool(gp & set(g["pmid"])))
        agg["gold_hit"] = hits
        agg["correct"] = agg["pred"] == agg["gold"]

        d = {}
        has = agg[agg.gold_hit.notna()]
        d["claims_with_gold_pmid"] = int(len(has))
        d["gold_in_top3"] = int(has.gold_hit.sum())
        d["acc_when_gold_retrieved"] = float(
            has[has.gold_hit == True].correct.mean())  # noqa: E712
        miss = has[has.gold_hit == False]  # noqa: E712
        d["gold_missed"] = int(len(miss))
        d["acc_when_gold_missed"] = (float(miss.correct.mean())
                                     if len(miss) else None)
        nei = agg[agg.gold == NEI]
        d["nei_claims"] = int(len(nei))
        d["nei_claims_predicted_non_nei"] = int((nei.pred != NEI).sum())
        # per-abstract view for NEI claims
        nei_rows = df[df.gold_claim_label == NEI]
        d["nei_abstract_votes_non_nei"] = float(
            (nei_rows[COLS].to_numpy().argmax(1) != NEI).mean())
        d["variants"] = {
            k: {"macro_f1": classification_metrics(v.gold, v.pred)["macro_f1"],
                "accuracy": classification_metrics(v.gold, v.pred)["accuracy"]}
            for k, v in agg_variants(df).items()}
        res[s] = d

        lines += [f"## {s}",
                  f"- claims with a gold PMID: {d['claims_with_gold_pmid']}; "
                  f"gold in top-3: {d['gold_in_top3']}, "
                  f"missed: {d['gold_missed']}",
                  f"- accuracy when gold retrieved: "
                  f"{d['acc_when_gold_retrieved']:.3f}",
                  f"- accuracy when gold missed: {d['acc_when_gold_missed']}",
                  f"- NEI claims: {d['nei_claims']}; predicted non-NEI: "
                  f"{d['nei_claims_predicted_non_nei']}",
                  f"- share of single abstracts of NEI claims voted non-NEI: "
                  f"{d['nei_abstract_votes_non_nei']:.3f}",
                  "- aggregation variants (macro F1 / acc):"]
        for k, v in d["variants"].items():
            lines.append(f"  - {k}: {v['macro_f1']:.3f} / "
                         f"{v['accuracy']:.3f}")
        lines.append("")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "e2_diagnostic.json").write_text(json.dumps(res, indent=2))
    (OUT_DIR / "e2_diagnostic.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
