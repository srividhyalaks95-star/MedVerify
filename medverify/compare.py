"""Paired comparison of all systems on the same claims (local, no API calls).

    python -m medverify.compare --split validation

Systems: B0, B0-knowledge, E2 top-1, E3 (frozen config), B0+correction.
Reports macro F1 / accuracy and paired bootstrap CIs of macro-F1 differences
(same resampled claims for both systems). Claims are matched by claim text.
E4 is selective (abstains), so it is not part of this full-coverage table.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from .config import LABELS, ROOT
from .e3_pipeline import OUT as E3_OUT, load as e3_load, predict as e3_predict
from .e3_pipeline import top1

LLM_OUT = ROOT / "results" / "llm"
IDS = sorted(LABELS)


def mf1(y, p):
    return float(f1_score(y, p, labels=IDS, average="macro", zero_division=0))


def main(split: str, n_boot: int = 2000, seed: int = 0):
    cfg = json.loads((E3_OUT / "chosen_config.json").read_text())
    cfg = (cfg["tau"], cfg["alpha"], cfg["fallback"], False)
    claims = e3_load(split)
    base = pd.DataFrame({"claim": [c["claim"] for c in claims],
                         "gold": [c["gold"] for c in claims],
                         "unseen": [c["unseen"] for c in claims],
                         "E2 top-1": [top1(c) for c in claims],
                         "E3": [e3_predict(c, cfg) for c in claims]})
    run = pd.read_csv(LLM_OUT / f"predictions_{split}_live_v2.csv")
    b0k = pd.read_csv(LLM_OUT / f"predictions_b0k_{split}_live.csv")
    df = base.merge(run[["claim", "b0", "cor"]], on="claim") \
             .merge(b0k[["claim", "b0k"]], on="claim")
    assert len(df) == len(base), "claim alignment lost rows"
    df = df.rename(columns={"b0": "B0", "b0k": "B0-knowledge",
                            "cor": "B0+correction"})
    names = ["B0", "B0-knowledge", "E2 top-1", "E3", "B0+correction"]
    y = df.gold.to_numpy()
    P = {n: df[n].to_numpy() for n in names}

    rng = np.random.default_rng(seed)
    idx = [rng.integers(0, len(y), len(y)) for _ in range(n_boot)]
    boot = {n: np.array([mf1(y[i], P[n][i]) for i in idx]) for n in names}

    out = {"split": split, "n": len(df), "systems": {}, "paired": {}}
    lines = [f"# System comparison ({split}, n={len(df)})", "",
             "| System | Macro F1 | 95% CI | Acc | Unseen macro F1 |",
             "|---|---|---|---|---|"]
    u = df[df.unseen]
    for n in names:
        lo, hi = np.percentile(boot[n], [2.5, 97.5])
        r = {"macro_f1": mf1(y, P[n]), "ci95": [float(lo), float(hi)],
             "acc": float((P[n] == y).mean()),
             "unseen_macro_f1": mf1(u.gold, u[n])}
        out["systems"][n] = r
        lines.append(f"| {n} | {r['macro_f1']:.3f} | {lo:.3f}-{hi:.3f} | "
                     f"{r['acc']:.3f} | {r['unseen_macro_f1']:.3f} |")
    lines += ["", "Paired difference in macro F1 (A - B):", "",
              "| A vs B | diff | 95% CI | excludes 0 |", "|---|---|---|---|"]
    for a, b in [("B0+correction", "E2 top-1"), ("B0+correction", "E3"),
                 ("B0+correction", "B0"), ("B0+correction", "B0-knowledge"),
                 ("E3", "E2 top-1")]:
        d = boot[a] - boot[b]
        lo, hi = np.percentile(d, [2.5, 97.5])
        pt = mf1(y, P[a]) - mf1(y, P[b])
        out["paired"][f"{a} - {b}"] = {"diff": pt, "ci95": [float(lo),
                                                            float(hi)]}
        lines.append(f"| {a} vs {b} | {pt:+.3f} | {lo:+.3f}..{hi:+.3f} | "
                     f"{'yes' if lo > 0 or hi < 0 else 'no'} |")
    ROOT.joinpath("results", "compare").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "compare" / f"compare_{split}.json").write_text(
        json.dumps(out, indent=2))
    (ROOT / "results" / "compare" / f"compare_{split}.md").write_text(
        "\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="validation")
    main(ap.parse_args().split)
