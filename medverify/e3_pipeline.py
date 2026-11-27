"""E3: context-aware evidence verification (seed-42 top-3 E2 scores).

    python -m medverify.e3_pipeline tune   # validation only -> chosen_config.json
    python -m medverify.e3_pipeline test   # loads the frozen config, runs test

Per claim: parse the template, score each of the 3 retrieved abstracts for
context alignment (intervention / comparator / outcome coverage), let only
abstracts with alignment >= tau vote, weight votes by alignment**alpha,
optionally map a split between Evidence and Inconclusive votes to
Inconclusive, and fall back (NEI or the rank-1 prediction) when no abstract
matches. Population is not in the claim template and is not used.
Compared against E2 top-1 with paired bootstrap CIs.
"""
from __future__ import annotations

import itertools
import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from .config import LABELS
from .context import alignment, parse_claim
from .e2_pipeline import E2_DIR, NEI, ROOT
from .metrics import classification_metrics
from .retrieval import _read

OUT = ROOT / "results" / "e3"
IDS = sorted(LABELS)
INCONCL = next(k for k, v in LABELS.items() if v == "Inconclusive")
COLS = [f"prob_{LABELS[i]}" for i in IDS]
GRID = list(itertools.product([0.0, 0.3, 0.4, 0.5, 0.6, 0.7], [0, 1, 2],
                              ["nei", "top1"], [True, False]))
N_BOOT = 2000
SEED = 42


def load(split: str) -> list[dict]:
    path = sorted(E2_DIR.glob(f"scores_*_{split}.csv"))[0]
    df = pd.read_csv(path, dtype={"pmid": str})
    abstract = {d["pmid"]: d["abstract"] for d in _read("corpus.jsonl")}
    claims = []
    for claim, g in df.groupby("claim", sort=False):
        g = g.sort_values("rank")
        parsed = parse_claim(claim)
        a = []
        for pm in g["pmid"]:
            if parsed is None:
                a.append(1.0)
                continue
            f = [v for v in alignment(parsed, abstract[pm]).values()
                 if v is not None]
            a.append(float(np.mean(f)) if f else 1.0)
        claims.append({"claim": claim, "p": g[COLS].to_numpy(),
                       "a": np.array(a), "gold": int(g.gold_claim_label.iloc[0]),
                       "unseen": bool(g.claim_unseen.iloc[0]),
                       "run": path.name})
    return claims


def predict(c: dict, cfg) -> int:
    tau, alpha, fb, split = cfg
    p, a = c["p"], c["a"]
    m = [i for i in range(len(a)) if a[i] >= tau]
    if not m:
        return NEI if fb == "nei" else int(p[0].argmax())
    w = a[m] ** alpha
    if w.sum() == 0:
        w = np.ones(len(m))
    pred = int(((w[:, None] * p[m]).sum(0) / w.sum()).argmax())
    if split:
        votes = {int(p[i].argmax()) for i in m} - {NEI}
        if len(votes) >= 2:
            pred = INCONCL
    return pred


def top1(c: dict) -> int:
    return int(c["p"][0].argmax())


def macro(gold, pred) -> float:
    return float(f1_score(gold, pred, labels=IDS, average="macro",
                          zero_division=0))


def tune():
    val = load("validation")
    gold = np.array([c["gold"] for c in val])
    rows = []
    for cfg in GRID:
        pred = np.array([predict(c, cfg) for c in val])
        rows.append({"tau": cfg[0], "alpha": cfg[1], "fallback": cfg[2],
                     "split_to_inconclusive": cfg[3],
                     "val_macro_f1": macro(gold, pred),
                     "val_acc": float((pred == gold).mean())})
    res = pd.DataFrame(rows).sort_values("val_macro_f1", ascending=False,
                                         kind="stable")
    best = res[~res["split_to_inconclusive"]].iloc[0].to_dict()  # primary
    abl = res[res["split_to_inconclusive"]].iloc[0].to_dict()    # ablation
    ref = macro(gold, np.array([top1(c) for c in val]))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "chosen_config.json").write_text(json.dumps(
        {"tau": best["tau"], "alpha": int(best["alpha"]),
         "fallback": best["fallback"],
         "split_to_inconclusive": False,
         "val_macro_f1": best["val_macro_f1"],
         "val_macro_f1_e2_top1": ref, "grid_size": len(GRID),
         "ablation_split_on": {
             "tau": abl["tau"], "alpha": int(abl["alpha"]),
             "fallback": abl["fallback"],
             "val_macro_f1": abl["val_macro_f1"]}}, indent=2))
    res.to_csv(OUT / "val_grid.csv", index=False)
    print(f"validation claims: {len(val)}; grid size: {len(GRID)}")
    print(f"E2 top-1 val macro F1: {ref:.3f}")
    print("top 8 configs (validation only):")
    print(res.head(8).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("chosen:", {k: best[k] for k in
                      ("tau", "alpha", "fallback", "split_to_inconclusive")})


def boot_ci(gold, pa, pb):
    rng = np.random.default_rng(SEED)
    n = len(gold)
    da, db, dd = [], [], []
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        fa, fb_ = macro(gold[i], pa[i]), macro(gold[i], pb[i])
        da.append(fa)
        db.append(fb_)
        dd.append(fa - fb_)
    q = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
    return {"e3_ci": q(da), "e2_ci": q(db), "diff_ci": q(dd),
            "diff_mean": float(np.mean(dd))}


def test():
    cfgd = json.loads((OUT / "chosen_config.json").read_text())
    cfg = (cfgd["tau"], cfgd["alpha"], cfgd["fallback"],
           cfgd["split_to_inconclusive"])
    te = load("test")
    gold = np.array([c["gold"] for c in te])
    p3 = np.array([predict(c, cfg) for c in te])
    p2 = np.array([top1(c) for c in te])
    unseen = np.array([c["unseen"] for c in te])
    res = {"config": cfgd, "run": te[0]["run"], "n": len(te)}
    for name, p in (("e3", p3), ("e2_top1", p2)):
        res[name] = {"all": classification_metrics(gold, p),
                     "unseen": classification_metrics(gold[unseen], p[unseen])}
    res["bootstrap_all"] = boot_ci(gold, p3, p2)
    res["bootstrap_unseen"] = boot_ci(gold[unseen], p3[unseen], p2[unseen])
    pd.DataFrame({"claim": [c["claim"] for c in te], "gold": gold,
                  "e2_top1": p2, "e3": p3, "unseen": unseen}
                 ).to_csv(OUT / "test_predictions.csv", index=False)
    (OUT / "e3_results.json").write_text(json.dumps(res, indent=2))

    order = [LABELS[i] for i in IDS]
    L = [f"E3 config (frozen from validation): {cfgd}", "",
         "| system | subset | n | macro F1 | acc |", "|---|---|---|---|---|"]
    for name in ("e2_top1", "e3"):
        for sub in ("all", "unseen"):
            m = res[name][sub]
            L.append(f"| {name} | {sub} | {m['n']} | {m['macro_f1']:.3f} | "
                     f"{m['accuracy']:.3f} |")
    for key in ("bootstrap_all", "bootstrap_unseen"):
        b = res[key]
        L.append(f"\n{key}: E3 CI {b['e3_ci']}, E2 CI {b['e2_ci']}, "
                 f"diff (E3-E2) mean {b['diff_mean']:.3f}, "
                 f"95% CI {b['diff_ci']}")
    L.append(f"\nE3 test confusion (rows=true, cols=pred, order {order}):")
    L.append(str(res["e3"]["all"]["confusion_matrix"]))
    (OUT / "e3_results.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    {"tune": tune, "test": test}[sys.argv[1]]()
