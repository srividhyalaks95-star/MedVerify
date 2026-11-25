"""Step 3: baseline evaluation report for E1.

Reads the saved run folders and writes results/baseline_report.md and
results/baseline_report.json. Run from Final_Project/:

    python -m medverify.report_baseline
    python -m medverify.report_baseline --runs results/e1_biobert_head_seed42 \
        results/e1_biobert_head_tail_seed42

Rules:
- The winning truncation policy is chosen on VALIDATION macro F1 only.
- Test numbers are reported for every run but never used to choose.
- The majority-class baseline predicts the most frequent TRAIN label.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import LABELS, RESULTS_DIR
from .data import load_all
from .metrics import classification_metrics

DEFAULT_RUNS = ("e1_biobert_head_seed42", "e1_biobert_head_tail_seed42")
NAMES = [LABELS[i] for i in sorted(LABELS)]


def load_run(run_dir: Path) -> dict:
    cfg = json.loads((run_dir / "config.json").read_text())
    met = json.loads((run_dir / "metrics.json").read_text())
    preds = {}
    for split, fname in (("validation", "val_predictions.csv"),
                         ("test", "test_predictions.csv")):
        df = pd.read_csv(run_dir / fname)
        df["claim_unseen"] = df["claim_unseen"].astype(bool)
        preds[split] = df
    return {"dir": run_dir, "config": cfg, "metrics": met, "preds": preds,
            "policy": cfg["args"]["policy"]}


def subset_metrics(df: pd.DataFrame) -> dict:
    out = {"all": classification_metrics(df["label"], df["pred"])}
    unseen = df[df["claim_unseen"]]
    out["claim_unseen"] = (classification_metrics(unseen["label"],
                                                  unseen["pred"])
                           if len(unseen) else None)
    return out


def majority_baseline() -> dict:
    data = load_all()
    majority = int(data["train"]["label"].value_counts().idxmax())
    res = {"majority_label": majority, "majority_name": LABELS[majority]}
    for split in ("validation", "test"):
        y = data[split]["label"].values
        res[split] = classification_metrics(y, np.full_like(y, majority))
    return res


def fmt(x: float) -> str:
    return f"{x:.3f}"


def per_class_table(m: dict) -> list[str]:
    rows = ["| Class | Precision | Recall | F1 | Support |",
            "|---|---|---|---|---|"]
    for name in NAMES:
        c = m["per_class"][name]
        rows.append(f"| {name} | {fmt(c['precision'])} | {fmt(c['recall'])} | "
                    f"{fmt(c['f1'])} | {c['support']} |")
    return rows


def confusion_table(m: dict) -> list[str]:
    rows = ["| true \\ pred | " + " | ".join(NAMES) + " |",
            "|---|" + "---|" * len(NAMES)]
    for name, r in zip(NAMES, m["confusion_matrix"]):
        rows.append(f"| **{name}** | " + " | ".join(str(v) for v in r) + " |")
    return rows


def best_epoch(run: dict) -> dict:
    evals = [e for e in run["metrics"]["log_history"] if "eval_macro_f1" in e]
    if not evals:
        return {}
    b = max(evals, key=lambda e: e["eval_macro_f1"])
    return {"epoch": b.get("epoch"), "val_macro_f1": b["eval_macro_f1"]}


def build_report(runs: list[dict], base: dict) -> tuple[str, dict]:
    # Choose the winner on validation macro F1 only
    val_f1 = {r["policy"]: r["metrics"]["validation"]["macro_f1"]
              for r in runs}
    winner = max(val_f1, key=val_f1.get)

    L: list[str] = ["# E1 baseline report (BioBERT claim + abstract)", ""]
    L += ["Scope: CliniFact clinical-research claim verification with "
          "pre-paired abstracts. Not general medical QA.", ""]
    L += [f"Labels: " + ", ".join(f"{k}={v}" for k, v in LABELS.items()), ""]

    # 1. Headline
    L += ["## 1. Headline: macro F1 and accuracy", "",
          "| System | Val macro F1 | Val acc | Test macro F1 | Test acc |",
          "|---|---|---|---|---|"]
    bv, bt = base["validation"], base["test"]
    L.append(f"| Majority class ({base['majority_name']}) | "
             f"{fmt(bv['macro_f1'])} | {fmt(bv['accuracy'])} | "
             f"{fmt(bt['macro_f1'])} | {fmt(bt['accuracy'])} |")
    for r in runs:
        v, t = r["metrics"]["validation"], r["metrics"]["test"]
        tag = " (selected)" if r["policy"] == winner else ""
        L.append(f"| BioBERT `{r['policy']}`{tag} | {fmt(v['macro_f1'])} | "
                 f"{fmt(v['accuracy'])} | {fmt(t['macro_f1'])} | "
                 f"{fmt(t['accuracy'])} |")
    L += ["", f"Selected policy: **`{winner}`**, chosen on validation macro F1 "
          "only. Test numbers were not used for selection.", ""]

    # 2. Truncation comparison
    L += ["## 2. Truncation comparison (validation)", "",
          "| Policy | Macro F1 | " + " | ".join(f"F1 {n}" for n in NAMES) +
          " | Best epoch |", "|---|---|" + "---|" * len(NAMES) + "---|"]
    for r in runs:
        v = r["metrics"]["validation"]
        be = best_epoch(r)
        L.append(f"| `{r['policy']}` | {fmt(v['macro_f1'])} | " + " | ".join(
            fmt(v["per_class"][n]["f1"]) for n in NAMES) +
            f" | {be.get('epoch', '?')} |")
    if len(runs) == 2:
        d = abs(val_f1[runs[0]["policy"]] - val_f1[runs[1]["policy"]])
        L += ["", f"Absolute validation macro-F1 gap: {fmt(d)}. With 316 "
              "validation rows and a single seed, treat this difference as "
              "tentative until Step 12 adds bootstrap intervals and extra "
              "seeds.", ""]

    # 3. Per-run details
    for r in runs:
        L += [f"## 3. Run detail: `{r['policy']}`", ""]
        for split in ("validation", "test"):
            sm = subset_metrics(r["preds"][split])
            L += [f"### {split}: all rows (n={sm['all']['n']})", "",
                  f"Accuracy {fmt(sm['all']['accuracy'])}, macro F1 "
                  f"{fmt(sm['all']['macro_f1'])}", ""]
            L += per_class_table(sm["all"]) + [""]
            L += ["Confusion matrix:", ""] + confusion_table(sm["all"]) + [""]
            cu = sm["claim_unseen"]
            if cu:
                L += [f"### {split}: claim-unseen subset (n={cu['n']})", "",
                      f"Accuracy {fmt(cu['accuracy'])}, macro F1 "
                      f"{fmt(cu['macro_f1'])}", ""]
                L += per_class_table(cu) + [""]

    # 4. Claim-unseen gap
    L += ["## 4. Memorization check: all rows vs claim-unseen (test)", "",
          "| Policy | Macro F1 (all) | Macro F1 (claim-unseen) | Gap |",
          "|---|---|---|---|"]
    for r in runs:
        sm = subset_metrics(r["preds"]["test"])
        a = sm["all"]["macro_f1"]
        u = sm["claim_unseen"]["macro_f1"] if sm["claim_unseen"] else float("nan")
        L.append(f"| `{r['policy']}` | {fmt(a)} | {fmt(u)} | {fmt(a - u)} |")
    L += ["", "A large positive gap would suggest the model benefits from "
          "claim text seen in training.", ""]

    # 5. Reproducibility
    L += ["## 5. Reproducibility", ""]
    for r in runs:
        c = r["config"]
        L.append(f"- `{r['policy']}`: seed {c['args']['seed']}, lr "
                 f"{c['args']['lr']}, epochs {c['args']['epochs']}, "
                 f"effective batch {c['effective_batch_size']}, device "
                 f"{c['device']}, transformers "
                 f"{c['packages'].get('transformers')}, torch "
                 f"{c['packages'].get('torch')}")
    L.append("")

    summary = {
        "selected_policy": winner,
        "validation_macro_f1": val_f1,
        "majority_baseline": base,
        "runs": {r["policy"]: {
            s: subset_metrics(r["preds"][s]) for s in ("validation", "test")}
            for r in runs},
    }
    return "\n".join(L), summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=None,
                    help="run folders (default: the two full E1 runs)")
    ap.add_argument("--out_dir", default=str(RESULTS_DIR))
    args = ap.parse_args()

    dirs = ([Path(p) for p in args.runs] if args.runs
            else [RESULTS_DIR / n for n in DEFAULT_RUNS])
    runs = [load_run(d) for d in dirs]
    base = majority_baseline()
    md, summary = build_report(runs, base)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "baseline_report.md").write_text(md)
    (out / "baseline_report.json").write_text(json.dumps(summary, indent=2))
    print(md)
    print(f"\nWrote {out / 'baseline_report.md'}")


if __name__ == "__main__":
    main()
