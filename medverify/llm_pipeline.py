"""B0 (claim only) and correction (claim + retrieved evidence).

    python -m medverify.llm_pipeline sample --n 8 --mode mock
    python -m medverify.llm_pipeline sample --n 8 --mode live

B0         : the LLM gives a verdict from the claim ALONE (no retrieval).
Correction : the LLM sees its own B0 draft plus the top-3 retrieved abstracts
             and must return a final verdict, grounded in the abstracts, and
             say whether it changed the draft. No verifier (E1-E4) output is
             shown to it, so B0 and correction measure only LLM +/- evidence.

Label names follow the project (Inconclusive / Evidence / NEI). The one-line
label definitions in the prompt are ASSUMPTIONS written for this project;
they are not copied from CliniFact documentation. Nothing here reads or
modifies E1-E4 results; only the validation split is used for the sample.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from .config import LABELS, ROOT
from .e2_pipeline import E2_DIR
from .llm import LLM
from .retrieval import _read

OUT = ROOT / "results" / "llm"
VERDICTS = [LABELS[i] for i in sorted(LABELS)]
MAX_ABS_CHARS = 3000

DEFS = ("Evidence: the trial results support the claimed comparison "
        "(the stated difference or equivalence is shown). "
        "Inconclusive: results exist for this outcome but do not establish "
        "the claimed comparison. "
        "NEI (not enough information): the available text does not report "
        "this outcome for this comparison.")

B0_SYSTEM = ("You verify claims about clinical trial outcomes. " + DEFS +
             " You have NO access to any paper: answer from the claim text "
             "alone and be honest about uncertainty. Reply in JSON.")
B0_SCHEMA = {"type": "object", "additionalProperties": False,
             "properties": {"verdict": {"type": "string", "enum": VERDICTS},
                            "reason": {"type": "string"}},
             "required": ["verdict", "reason"]}

COR_SYSTEM = ("You check a draft verdict about a clinical trial claim against "
              "retrieved abstracts. " + DEFS +
              " Use only the abstracts provided. Keep the draft if the "
              "abstracts do not justify a change; change it only when the "
              "abstracts clearly support another label. If no abstract "
              "reports the claimed outcome and comparison, answer NEI. "
              "Reply in JSON.")
COR_SCHEMA = {"type": "object", "additionalProperties": False,
              "properties": {
                  "verdict": {"type": "string", "enum": VERDICTS},
                  "changed_draft": {"type": "boolean"},
                  "evidence_pmid": {"type": "string"},
                  "reason": {"type": "string"}},
              "required": ["verdict", "changed_draft", "evidence_pmid",
                           "reason"]}


# ---- v2 prompts (v1 above is kept unchanged for before/after comparison) ----
# Definitions follow how CliniFact derives labels (ds4dh/CliniFact scripts
# c_extract_stance_label.py, e_evidence_labels4primary_outcome.py):
#  - stance is 'positive' if any reported p-value for the outcome is <= 0.05,
#    otherwise 'negative';
#  - Evidence = the trial's result paper has a positive stance;
#  - Inconclusive = the result paper exists but stance is negative (this
#    mapping was read from a truncated script: treated as confirmed-by-
#    inference);
#  - NEI = the paper is only a background reference, no results reported.
DEFS_V2 = (
    "Label definitions (from how the CliniFact dataset was built). "
    "Evidence: the trial's results paper reports a statistically significant "
    "result (p <= 0.05) for this outcome, i.e. the claimed comparison is "
    "demonstrated. "
    "Inconclusive: a results paper for this trial reports the outcome but the "
    "result is NOT statistically significant (p > 0.05), so the claimed "
    "difference is not demonstrated (e.g. 'no significant difference', "
    "'failed to show superiority', 'did not reach significance'). "
    "NEI (not enough information): no retrieved text reports results for "
    "this trial and outcome (only background or unrelated papers). "
    "Absence of evidence is NEI, never Evidence. Failing to show a claimed "
    "superiority is Inconclusive, never Evidence.")

B0_SYSTEM_V2 = ("You verify claims about clinical trial outcomes. " + DEFS_V2 +
                " You have NO access to any paper: answer from the claim "
                "text alone. Without a paper, answer NEI unless the claim is "
                "well-known. The verdict must agree with your reason. "
                "Reply in JSON.")
COR_SYSTEM_V2 = (
    "You check a draft verdict about a clinical trial claim against "
    "retrieved abstracts. " + DEFS_V2 +
    " Use only the abstracts. First decide whether an abstract is about the "
    "SAME trial: same intervention, comparator and outcome as the claim. "
    "Abstracts about related drugs, other populations or other outcomes must "
    "be ignored; if none matches, answer NEI. If a matching abstract "
    "reports a non-significant or failed comparison, answer Inconclusive; "
    "answer Evidence only if it reports a significant result for the "
    "claimed comparison. Do not change the draft without such an abstract. "
    "The verdict must agree with your reason. Reply in JSON.")

VERSION = "v1"

# B0-knowledge: fairer claim-only baseline. Same label definitions, but the
# model must commit to its best guess from medical knowledge (no NEI default).
B0_SYSTEM_K = (
    "You verify claims about clinical trial outcomes. " + DEFS_V2 +
    " You have NO access to any paper. Use your medical knowledge of the "
    "trial, drugs and typical results to predict the most likely label. "
    "Choose NEI only if you truly cannot guess. The verdict must agree with "
    "your reason. Reply in JSON.")

def _sys(kind: str) -> str:
    v2 = VERSION == "v2"
    return (B0_SYSTEM_V2 if v2 else B0_SYSTEM) if kind == "b0" else (
        COR_SYSTEM_V2 if v2 else COR_SYSTEM)


def b0(llm: LLM, claim: str) -> dict:
    return llm.call("b0", _sys("b0"), f"Claim: {claim}", B0_SCHEMA, "b0")


def correct(llm: LLM, claim: str, draft: str, abstracts: list[dict]) -> dict:
    ev = "\n\n".join(f"[PMID {a['pmid']}]\n{a['abstract'][:MAX_ABS_CHARS]}"
                     for a in abstracts)
    user = (f"Claim: {claim}\nDraft verdict: {draft}\n\n"
            f"Retrieved abstracts:\n{ev}\n\n"
            "Return the final verdict. Use evidence_pmid '' if none applies.")
    return llm.call("correction", _sys("cor"), user, COR_SCHEMA, "correction")


def load_claims(split: str) -> list[dict]:
    df = pd.read_csv(sorted(E2_DIR.glob(f"scores_*_{split}.csv"))[0],
                     dtype={"pmid": str})
    abstract = {d["pmid"]: d["abstract"] for d in _read("corpus.jsonl")}
    out = []
    for claim, g in df.groupby("claim", sort=False):
        g = g.sort_values("rank")
        out.append({"claim": claim, "gold": int(g.gold_claim_label.iloc[0]),
                    "unseen": bool(g.claim_unseen.iloc[0]),
                    "abstracts": [{"pmid": p, "abstract": abstract[p]}
                                  for p in g["pmid"]]})
    return out


def sample(n: int, mode: str, split: str = "validation", seed: int = 42):
    claims = load_claims(split)
    rng = np.random.default_rng(seed)
    # small stratified smoke sample (gold used only to pick varied examples)
    by = {i: [c for c in claims if c["gold"] == i] for i in sorted(LABELS)}
    pick, per = [], max(1, n // len(by))
    for i in sorted(by):
        idx = rng.permutation(len(by[i]))[:per]
        pick += [by[i][j] for j in idx]
    pick = pick[:n]

    llm = LLM(mode=mode)
    rows = []
    for c in pick:
        r0 = b0(llm, c["claim"])
        r1 = correct(llm, c["claim"], r0["verdict"], c["abstracts"])
        rows.append({"claim": c["claim"][:110],
                     "gold": LABELS[c["gold"]],
                     "b0": r0["verdict"], "b0_reason": r0["reason"],
                     "corrected": r1["verdict"],
                     "changed": r1["changed_draft"],
                     "evidence_pmid": r1["evidence_pmid"],
                     "cor_reason": r1["reason"],
                     "b0_tok": (r0["_usage"]["in_tok"],
                                r0["_usage"]["out_tok"]),
                     "cor_tok": (r1["_usage"]["in_tok"],
                                 r1["_usage"]["out_tok"])})
    OUT.mkdir(parents=True, exist_ok=True)
    summ = llm.summary()
    name = f"smoke_{mode}.json" if VERSION == "v1" else \
        f"smoke_{mode}_{VERSION}.json"
    (OUT / name).write_text(
        json.dumps({"rows": rows, "usage": summ}, indent=2))
    for r in rows:
        print(f"\nCLAIM: {r['claim']}")
        print(f"  gold={r['gold']} | B0={r['b0']} | corrected={r['corrected']}"
              f" (changed={r['changed']}, pmid={r['evidence_pmid'] or '-'})")
        print(f"  B0 reason: {r['b0_reason']}")
        print(f"  correction reason: {r['cor_reason']}")
        print(f"  tokens in/out  B0={r['b0_tok']}  correction={r['cor_tok']}")
    print("\nUSAGE:", json.dumps(summ, indent=2))


def _retry(fn, *args, tries: int = 4):
    import time
    for t in range(tries):
        try:
            return fn(*args)
        except Exception as e:  # rate limit / transient network
            if t == tries - 1:
                raise
            time.sleep(2 ** t)
            print("retry after error:", type(e).__name__)


def _one(llm: LLM, c: dict) -> dict:
    r0 = _retry(b0, llm, c["claim"])
    r1 = _retry(correct, llm, c["claim"], r0["verdict"], c["abstracts"])
    u0, u1 = r0["_usage"], r1["_usage"]
    return {"claim": c["claim"], "gold": c["gold"], "unseen": c["unseen"],
            "b0": VERDICTS.index(r0["verdict"]),
            "cor": VERDICTS.index(r1["verdict"]),
            "changed": r1["changed_draft"], "evidence_pmid": r1["evidence_pmid"],
            "b0_reason": r0["reason"], "cor_reason": r1["reason"],
            "b0_in": u0["in_tok"], "b0_out": u0["out_tok"],
            "b0_lat": u0["latency_s"], "b0_cost": u0["cost_usd"],
            "cor_in": u1["in_tok"], "cor_out": u1["out_tok"],
            "cor_lat": u1["latency_s"], "cor_cost": u1["cost_usd"]}


def _boot(y, a, b=None, n=2000, seed=0):
    """95% CI of macro F1 of a (and of a-b if b is given)."""
    from sklearn.metrics import f1_score
    rng = np.random.default_rng(seed)
    y, a = np.asarray(y), np.asarray(a)
    b = None if b is None else np.asarray(b)
    ids = sorted(LABELS)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        f = f1_score(y[i], a[i], labels=ids, average="macro", zero_division=0)
        if b is not None:
            f -= f1_score(y[i], b[i], labels=ids, average="macro",
                          zero_division=0)
        vals.append(f)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def run(split: str, mode: str, workers: int, confirm_frozen: bool):
    from concurrent.futures import ThreadPoolExecutor
    from .metrics import classification_metrics
    if split == "test" and not confirm_frozen:
        raise SystemExit("test run needs --confirm_frozen (prompt frozen on "
                         "validation)")
    if split == "test" and VERSION != "v2":
        raise SystemExit("freeze a prompt version first")
    claims = load_claims(split)
    llm = LLM(mode=mode, cache_name=f"run_{VERSION}")
    with ThreadPoolExecutor(workers) as ex:
        rows = list(ex.map(lambda c: _one(llm, c), claims))
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"{split}_{mode}_{VERSION}"
    df.to_csv(OUT / f"predictions_{tag}.csv", index=False)

    res = {"split": split, "version": VERSION, "mode": mode, "systems": {}}
    for name, col in (("B0", "b0"), ("B0+correction", "cor")):
        m = classification_metrics(df.gold, df[col])
        m["macro_f1_ci95"] = _boot(df.gold, df[col])
        u = df[df.unseen]
        m["unseen"] = classification_metrics(u.gold, u[col])
        res["systems"][name] = m
    res["diff_cor_minus_b0_macro_f1_ci95"] = _boot(df.gold, df.cor, df.b0)
    flips = df[df.b0 != df.cor]
    res["changes"] = {
        "n_changed": int(len(flips)),
        "fixed": int(((flips.cor == flips.gold)).sum()),
        "broken": int(((flips.b0 == flips.gold)).sum()),
        "wrong_to_wrong": int(((flips.cor != flips.gold)
                               & (flips.b0 != flips.gold)).sum())}
    res["overhead"] = {
        "claims": len(df),
        "b0": {"calls": len(df), "in_tok": int(df.b0_in.sum()),
               "out_tok": int(df.b0_out.sum()),
               "cost_usd": round(float(df.b0_cost.sum()), 4)},
        "correction": {"calls": len(df), "in_tok": int(df.cor_in.sum()),
                       "out_tok": int(df.cor_out.sum()),
                       "cost_usd": round(float(df.cor_cost.sum()), 4)},
        "run_summary_this_session": llm.summary(),
        "note": "cost 0 for cached calls; per-claim latency is sum of "
                "per-call latency (wall time is lower with parallelism)"}
    (OUT / f"results_{tag}.json").write_text(json.dumps(res, indent=2))
    for name, m in res["systems"].items():
        print(f"{name}: macro F1 {m['macro_f1']:.3f} "
              f"CI {m['macro_f1_ci95'][0]:.3f}-{m['macro_f1_ci95'][1]:.3f} "
              f"acc {m['accuracy']:.3f}")
        for k, v in m["per_class"].items():
            print(f"   {k}: P {v['precision']:.3f} R {v['recall']:.3f} "
                  f"n={v['support']}")
        print("   confusion", m["confusion_matrix"])
    print("diff (cor - B0) macro F1 CI95:",
          res["diff_cor_minus_b0_macro_f1_ci95"])
    print("changes:", res["changes"])
    print("overhead:", json.dumps(res["overhead"], indent=2))


def run_b0k(split: str, mode: str, workers: int, confirm_frozen: bool):
    """B0-knowledge: claim-only baseline that must commit to a best guess."""
    from concurrent.futures import ThreadPoolExecutor
    from .metrics import classification_metrics
    if split == "test" and not confirm_frozen:
        raise SystemExit("test run needs --confirm_frozen")
    claims = load_claims(split)
    llm = LLM(mode=mode, cache_name="run_b0k")

    def one(c):
        r = _retry(lambda: llm.call("b0k", B0_SYSTEM_K, f"Claim: {c['claim']}",
                                    B0_SCHEMA, "b0k"))
        u = r["_usage"]
        return {"claim": c["claim"], "gold": c["gold"], "unseen": c["unseen"],
                "b0k": VERDICTS.index(r["verdict"]), "reason": r["reason"],
                "in": u["in_tok"], "out": u["out_tok"], "cost": u["cost_usd"]}

    with ThreadPoolExecutor(workers) as ex:
        df = pd.DataFrame(list(ex.map(one, claims)))
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"{split}_{mode}"
    df.to_csv(OUT / f"predictions_b0k_{tag}.csv", index=False)
    m = classification_metrics(df.gold, df.b0k)
    m["macro_f1_ci95"] = _boot(df.gold, df.b0k)
    m["overhead"] = {"calls": len(df), "in_tok": int(df["in"].sum()),
                     "out_tok": int(df.out.sum()),
                     "cost_usd": round(float(df.cost.sum()), 4)}
    (OUT / f"results_b0k_{tag}.json").write_text(json.dumps(m, indent=2))
    print(f"B0-knowledge: macro F1 {m['macro_f1']:.3f} "
          f"CI {m['macro_f1_ci95'][0]:.3f}-{m['macro_f1_ci95'][1]:.3f} "
          f"acc {m['accuracy']:.3f}")
    for k, v in m["per_class"].items():
        print(f"   {k}: P {v['precision']:.3f} R {v['recall']:.3f}")
    print("   confusion", m["confusion_matrix"])
    print("overhead:", m["overhead"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sample", "run", "b0k"])
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--confirm_frozen", action="store_true")
    ap.add_argument("--mode", choices=["mock", "live"], default="mock")
    ap.add_argument("--prompt_version", choices=["v1", "v2"], default="v1")
    a = ap.parse_args()
    VERSION = a.prompt_version
    if a.cmd == "run":
        run(a.split, a.mode, a.workers, a.confirm_frozen)
    elif a.cmd == "b0k":
        run_b0k(a.split, a.mode, a.workers, a.confirm_frozen)
    else:
        sample(a.n, a.mode)
