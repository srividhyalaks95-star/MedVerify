"""E4: calibrated abstention on top of E3 (seed-42 stored E2 scores).

    python -m medverify.e4_pipeline tune   # validation only -> chosen_config.json
    python -m medverify.e4_pipeline test   # frozen config, run once on test

Claim probability = E3's alignment-weighted mean of the top-3 abstract
probabilities (frozen E3 config). Temperature scaling (fit on validation)
calibrates confidence without changing any predicted label. Abstention score
= calibrated confidence - lam * disagreement, where disagreement is the share
of the 3 retrieved abstracts whose own vote differs from the final label.
lam is chosen by validation AURC; the threshold is the one that keeps the
largest coverage while reaching TARGET_SEL_ACC on validation answered claims.
Abstaining is a decision to route the claim to a human, not a label.
"""
from __future__ import annotations

import json
import sys

import numpy as np

from .config import LABELS
from .e2_pipeline import ROOT
from .e3_pipeline import IDS, load, macro
from .metrics import classification_metrics

OUT = ROOT / "results" / "e4"
E3_CFG = ROOT / "results" / "e3" / "chosen_config.json"
TARGET_SEL_ACC = 0.75   # design choice, fixed before looking at test
MIN_COVERAGE = 0.30
LAMS = [0.0, 0.1, 0.2, 0.3, 0.5]
TEMPS = np.round(np.arange(0.5, 5.01, 0.05), 2)
N_BOOT = 2000
SEED = 42


def claim_probs(c: dict, tau: float, alpha: float) -> np.ndarray:
    p, a = c["p"], c["a"]
    m = [i for i in range(len(a)) if a[i] >= tau]
    if not m:
        return p[0]
    w = a[m] ** alpha
    if w.sum() == 0:
        w = np.ones(len(m))
    return (w[:, None] * p[m]).sum(0) / w.sum()


def features(claims: list[dict], e3: dict):
    P, D = [], []
    for c in claims:
        p = claim_probs(c, e3["tau"], e3["alpha"])
        P.append(p)
        D.append(float((c["p"].argmax(1) != int(p.argmax())).mean()))
    return np.array(P), np.array(D)


def soft_t(p: np.ndarray, T: float) -> np.ndarray:
    z = np.log(np.clip(p, 1e-9, 1)) / T
    z -= z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def nll(p, gold) -> float:
    return float(-np.mean(np.log(np.clip(p[np.arange(len(gold)), gold],
                                         1e-9, 1))))


def ece(conf, correct, bins=10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    tot = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            tot += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(tot)


def aurc(score, correct) -> float:
    o = np.argsort(-score, kind="stable")
    err = 1 - correct[o]
    return float(np.mean(np.cumsum(err) / np.arange(1, len(err) + 1)))


def pick_thr(score, correct):
    best, fallback = None, None
    for t in np.unique(score):
        ans = score >= t
        cov = ans.mean()
        if cov < MIN_COVERAGE:
            continue
        sa = correct[ans].mean()
        if sa >= TARGET_SEL_ACC and (best is None or cov > best[1]):
            best = (float(t), float(cov), float(sa))
        if fallback is None or sa > fallback[2]:
            fallback = (float(t), float(cov), float(sa))
    return best, fallback


def tune():
    e3 = json.loads(E3_CFG.read_text())
    val = load("validation")
    gold = np.array([c["gold"] for c in val])
    P, D = features(val, e3)
    pred = P.argmax(1)
    correct = (pred == gold).astype(float)
    T = float(min(TEMPS, key=lambda t: nll(soft_t(P, t), gold)))
    conf = soft_t(P, T).max(1)
    per_lam = {}
    for lam in LAMS:
        s = conf - lam * D
        best, fb = pick_thr(s, correct)
        chosen = best or fb
        per_lam[str(lam)] = {"aurc": aurc(s, correct), "threshold": chosen[0],
                             "val_coverage": chosen[1],
                             "val_sel_acc": chosen[2],
                             "target_reached": best is not None}
    lam = min(LAMS, key=lambda l: (per_lam[str(l)]["aurc"], l))
    cfg = {"temperature": T, "lam": lam, "per_lam": per_lam,
           "target_sel_acc": TARGET_SEL_ACC, "min_coverage": MIN_COVERAGE,
           "val_n": len(val), "val_acc_full": float(correct.mean()),
           "val_ece_raw": ece(P.max(1), correct),
           "val_ece_cal": ece(conf, correct),
           "val_nll_raw": nll(P, gold), "val_nll_cal": nll(soft_t(P, T), gold)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "chosen_config.json").write_text(json.dumps(cfg, indent=2))
    print(json.dumps(cfg, indent=2))


def _sel(ans, correct):
    return float(correct[ans].mean()) if ans.any() else float("nan")


def test():
    cfg = json.loads((OUT / "chosen_config.json").read_text())
    e3 = json.loads(E3_CFG.read_text())
    te = load("test")
    gold = np.array([c["gold"] for c in te])
    unseen = np.array([c["unseen"] for c in te])
    P, D = features(te, e3)
    pred = P.argmax(1)
    correct = (pred == gold).astype(float)
    T, lam = cfg["temperature"], cfg["lam"]
    conf = soft_t(P, T).max(1)
    score = conf - lam * D
    thr = cfg["per_lam"][str(lam)]["threshold"]
    ans = score >= thr

    res = {"n": len(te), "config": {"T": T, "lam": lam, "threshold": thr},
           "acc_full": float(correct.mean()),
           "macro_f1_full": macro(gold, pred),
           "ece_raw": ece(P.max(1), correct), "ece_cal": ece(conf, correct),
           "nll_raw": nll(P, gold), "nll_cal": nll(soft_t(P, T), gold),
           "coverage": float(ans.mean()), "answered": int(ans.sum()),
           "abstained": int((~ans).sum()),
           "sel_acc": _sel(ans, correct),
           "sel_macro_f1": (macro(gold[ans], pred[ans]) if ans.any()
                            else None),
           "errors_abstained": int((~ans & (correct == 0)).sum()),
           "errors_total": int((correct == 0).sum()),
           "correct_abstained": int((~ans & (correct == 1)).sum()),
           "sel_acc_unseen": _sel(ans & unseen, correct),
           "coverage_unseen": float(ans[unseen].mean()),
           "abstain_by_true": {LABELS[i]: int((~ans & (gold == i)).sum())
                               for i in IDS},
           "n_by_true": {LABELS[i]: int((gold == i).sum()) for i in IDS},
           "abstain_by_pred": {LABELS[i]: int((~ans & (pred == i)).sum())
                               for i in IDS}}
    # selective confusion matrix on answered claims
    res["answered_confusion"] = classification_metrics(
        gold[ans], pred[ans])["confusion_matrix"]
    # risk-coverage at fixed coverage levels (rank by chosen score)
    o = np.argsort(-score, kind="stable")
    rc = {}
    for cov in (1.0, 0.8, 0.6, 0.4):
        k = int(round(cov * len(te)))
        rc[str(cov)] = float(correct[o[:k]].mean())
    res["acc_at_coverage"] = rc
    # ablation: confidence only (lam=0) with its own validation threshold
    s0 = conf
    t0 = cfg["per_lam"]["0.0"]["threshold"]
    a0 = s0 >= t0
    res["ablation_confidence_only"] = {
        "threshold": t0, "coverage": float(a0.mean()),
        "sel_acc": _sel(a0, correct), "aurc": aurc(s0, correct)}
    res["aurc"] = {"chosen_score": aurc(score, correct),
                   "confidence_only": aurc(s0, correct),
                   "random_abstention": float(1 - correct.mean())}
    # bootstrap CI for selective accuracy
    rng = np.random.default_rng(SEED)
    sa, cv = [], []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(te), len(te))
        if ans[i].any():
            sa.append(correct[i][ans[i]].mean())
            cv.append(ans[i].mean())
    res["sel_acc_ci"] = [float(np.percentile(sa, 2.5)),
                         float(np.percentile(sa, 97.5))]
    res["coverage_ci"] = [float(np.percentile(cv, 2.5)),
                          float(np.percentile(cv, 97.5))]
    (OUT / "e4_results.json").write_text(json.dumps(res, indent=2))

    L = [f"E4 frozen config: T={T}, lam={lam}, threshold={thr:.3f} "
         f"(target selective acc {cfg['target_sel_acc']} on validation)", "",
         f"- test claims: {res['n']}; accuracy without abstention: "
         f"{res['acc_full']:.3f}; macro F1: {res['macro_f1_full']:.3f}",
         f"- calibration (test): ECE {res['ece_raw']:.3f} -> "
         f"{res['ece_cal']:.3f}; NLL {res['nll_raw']:.3f} -> "
         f"{res['nll_cal']:.3f}",
         f"- coverage: {res['coverage']:.3f} ({res['answered']} answered, "
         f"{res['abstained']} abstained), 95% CI {res['coverage_ci']}",
         f"- accuracy on answered claims: {res['sel_acc']:.3f}, "
         f"95% CI {res['sel_acc_ci']}",
         f"- macro F1 on answered: {res['sel_macro_f1']}",
         f"- errors abstained: {res['errors_abstained']} of "
         f"{res['errors_total']}; correct answers abstained: "
         f"{res['correct_abstained']}",
         f"- unseen: coverage {res['coverage_unseen']:.3f}, "
         f"accuracy on answered {res['sel_acc_unseen']:.3f}",
         f"- abstained by true class: {res['abstain_by_true']} "
         f"(of {res['n_by_true']})",
         f"- abstained by predicted class: {res['abstain_by_pred']}",
         f"- accuracy at coverage (ranked by score): {rc}",
         f"- AURC (lower is better): {res['aurc']}",
         f"- ablation, confidence only (no disagreement): "
         f"{res['ablation_confidence_only']}",
         "", "Answered-claims confusion (rows=true, cols=pred, order "
         f"{[LABELS[i] for i in IDS]}):", str(res["answered_confusion"])]
    (OUT / "e4_results.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    {"tune": tune, "test": test}[sys.argv[1]]()
