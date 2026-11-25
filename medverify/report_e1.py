"""Step 3b: multi-seed E1 report.

Aggregates every full E1 run under results/ (folders e1_biobert_*_seed*,
smoke runs excluded) and writes results/e1_report.md and results/e1_report.json.

    python -m medverify.report_e1

Adds over report_baseline.py:
- mean and sd across seeds for each (truncation policy, class weighting) group
- head vs head_tail compared seed by seed
- effect of class weighting on how often Inconclusive is predicted
- seen and unseen claims reported separately, plus the repeated-claim label
  imbalance behind that split (the main limitation of this baseline)

Selection uses VALIDATION only. Test is reported, never used to choose.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .config import LABELS, RESULTS_DIR
from .data import load_all
from .metrics import classification_metrics
from .report_baseline import NAMES, fmt, load_run

NEI_ID = next(k for k, v in LABELS.items() if v == "NEI")
INC = LABELS[0]
SPLITS = ("validation", "test")


# ---------- loading and per-run statistics ----------
def discover(results_dir: Path) -> list[Path]:
    return sorted(
        p for p in results_dir.glob("e1_biobert_*_seed*")
        if p.is_dir() and "smoke" not in p.name
        and (p / "metrics.json").exists())


def run_key(run: dict) -> tuple[str, str]:
    weighted = not run["config"]["args"].get("no_class_weights", False)
    return run["policy"], "weighted" if weighted else "unweighted"


def seed_of(run: dict) -> int:
    return int(run["config"]["args"]["seed"])


def run_stats(run: dict) -> dict:
    nan = float("nan")
    out = {}
    for split in SPLITS:
        df = run["preds"][split]
        seen, unseen = df[~df["claim_unseen"]], df[df["claim_unseen"]]
        a = classification_metrics(df["label"], df["pred"])
        u = classification_metrics(unseen["label"], unseen["pred"])
        out[split] = {
            "macro_f1": a["macro_f1"],
            "acc": a["accuracy"],
            "precision": {n: a["per_class"][n]["precision"] for n in NAMES},
            "recall": {n: a["per_class"][n]["recall"] for n in NAMES},
            "f1": {n: a["per_class"][n]["f1"] for n in NAMES},
            "unseen_macro_f1": u["macro_f1"],
            "unseen_acc": u["accuracy"],
            "unseen_n": int(len(unseen)),
            "unseen_nei_share": (float((unseen["label"] == NEI_ID).mean())
                                 if len(unseen) else nan),
            "seen_n": int(len(seen)),
            "seen_acc": (float((seen["pred"] == seen["label"]).mean())
                         if len(seen) else nan),
            "seen_nei_share": (float((seen["label"] == NEI_ID).mean())
                               if len(seen) else nan),
            "pred_share": {LABELS[i]: float((df["pred"] == i).mean())
                           for i in sorted(LABELS)},
            "true_share": {LABELS[i]: float((df["label"] == i).mean())
                           for i in sorted(LABELS)},
        }
    return out


def pick(stats: list[dict], split: str, key: str, sub: str | None = None):
    return [s[split][key] if sub is None else s[split][key][sub]
            for s in stats]


def mean_sd(vals) -> str:
    v = np.asarray(vals, dtype=float)
    if len(v) == 0:
        return "n/a"
    if len(v) == 1:
        return f"{v[0]:.3f}"
    return f"{v.mean():.3f} ± {v.std(ddof=1):.3f}"


def label(key: tuple[str, str]) -> str:
    return f"`{key[0]}` / {key[1]}"


# ---------- data-side statistics ----------
def majority_rows(data: dict) -> tuple[str, dict]:
    maj = int(data["train"]["label"].value_counts().idxmax())
    out = {}
    for s in SPLITS:
        df = data[s]
        un = df[df["claim_unseen"]]
        a = classification_metrics(df["label"], np.full(len(df), maj))
        u = classification_metrics(un["label"], np.full(len(un), maj))
        out[s] = {"macro_f1": a["macro_f1"], "acc": a["accuracy"],
                  "unseen_macro_f1": u["macro_f1"]}
    return LABELS[maj], out


def repeated_claim_stats(data: dict) -> dict:
    a = pd.concat([data[s].assign(split=s) for s in data], ignore_index=True)
    a["claim_rows"] = a["claim"].map(a.groupby("claim").size())
    rep, one = a[a["claim_rows"] > 1], a[a["claim_rows"] == 1]
    tr = a[a["split"] == "train"]

    def nei(df):
        return float((df["label"] == NEI_ID).mean())

    g = rep.groupby("claim").agg(
        n_nct=("nctId", "nunique"),
        non_nei=("label", lambda s: int((s != NEI_ID).sum())))
    return {
        "rows": int(len(a)),
        "distinct_claims": int(a["claim"].nunique()),
        "repeated_claims": int(len(g)),
        "rows_in_repeated": int(len(rep)),
        "rows_unique": int(len(one)),
        "nei_repeated": nei(rep),
        "nei_unique": nei(one),
        "nei_repeated_train": nei(tr[tr["claim_rows"] > 1]),
        "nei_unique_train": nei(tr[tr["claim_rows"] == 1]),
        "claims_one_nct": int((g["n_nct"] == 1).sum()),
        "claims_no_non_nei": int((g["non_nei"] == 0).sum()),
    }


# ---------- report ----------
def build_report(groups: dict, data: dict) -> tuple[str, dict]:
    stats = {k: [r["stats"] for r in rs] for k, rs in groups.items()}
    best = max(groups, key=lambda k: np.mean(
        pick(stats[k], "validation", "macro_f1")))
    maj_name, maj = majority_rows(data)
    rc = repeated_claim_stats(data)

    L: list[str] = ["# E1 report: BioBERT claim + abstract (multi-seed)", ""]
    L += ["Scope: CliniFact claim verification with pre-paired abstracts "
          "(not general medical QA). 'Unseen' = claim text not present in "
          "train; 'seen' = it is.", ""]

    # 1. Headline
    L += ["## 1. Headline (mean ± sd over seeds)", "",
          "| System | Seeds | Val macro F1 | Val macro F1, unseen | "
          "Test macro F1 | Test macro F1, unseen | Test acc |",
          "|---|---|---|---|---|---|---|",
          f"| Majority ({maj_name}) | - | {fmt(maj['validation']['macro_f1'])}"
          f" | {fmt(maj['validation']['unseen_macro_f1'])} | "
          f"{fmt(maj['test']['macro_f1'])} | "
          f"{fmt(maj['test']['unseen_macro_f1'])} | "
          f"{fmt(maj['test']['acc'])} |"]
    for k, rs in groups.items():
        st = stats[k]
        L.append(
            f"| BioBERT {label(k)} | {len(rs)} | "
            f"{mean_sd(pick(st, 'validation', 'macro_f1'))} | "
            f"{mean_sd(pick(st, 'validation', 'unseen_macro_f1'))} | "
            f"{mean_sd(pick(st, 'test', 'macro_f1'))} | "
            f"{mean_sd(pick(st, 'test', 'unseen_macro_f1'))} | "
            f"{mean_sd(pick(st, 'test', 'acc'))} |")
    L += ["", f"Best group by mean validation macro F1: **{label(best)}**. "
          "Test numbers were not used for selection.", ""]
    if any(len(rs) < 3 for rs in groups.values()):
        L += ["Note: some groups have fewer than 3 seeds; sd is "
              "unavailable or unreliable.", ""]

    # 2. head vs head_tail
    L += ["## 2. head vs head_tail, seed by seed (head_tail minus head)", ""]
    paired_any = False
    wins: dict = {}
    for w in ("weighted", "unweighted"):
        h = {seed_of(r): r["stats"] for r in groups.get(("head", w), [])}
        t = {seed_of(r): r["stats"] for r in groups.get(("head_tail", w), [])}
        seeds = sorted(set(h) & set(t))
        if not seeds:
            continue
        paired_any = True
        cols = [("validation", "macro_f1", "Δ val macro F1"),
                ("validation", "unseen_macro_f1", "Δ val unseen"),
                ("test", "macro_f1", "Δ test macro F1"),
                ("test", "unseen_macro_f1", "Δ test unseen")]
        L += [f"### {w}", "",
              "| Seed | " + " | ".join(c[2] for c in cols) + " |",
              "|---|" + "---|" * len(cols)]
        diffs = {c[2]: [] for c in cols}
        for s in seeds:
            row = []
            for split, key, name in cols:
                d = t[s][split][key] - h[s][split][key]
                diffs[name].append(d)
                row.append(f"{d:+.3f}")
            L.append(f"| {s} | " + " | ".join(row) + " |")
        L.append("| **wins** | " + " | ".join(
            f"{sum(d > 0 for d in diffs[c[2]])}/{len(seeds)}"
            for c in cols) + " |")
        L.append("")
        wins[w] = {n: [float(x) for x in v] for n, v in diffs.items()}
    if not paired_any:
        L += ["No seed has both policies yet.", ""]

    # 3. class weighting
    L += ["## 3. Effect of class weighting on Inconclusive", "",
          "Compare the predicted share of Inconclusive with its true share. "
          "A predicted share well above the true share means the model "
          "over-predicts Inconclusive.", ""]
    for split in SPLITS:
        L += [f"### {split}", "",
              "| System | Inconclusive predicted | true | precision | "
              "recall | F1 | Macro F1 |", "|---|---|---|---|---|---|---|"]
        for k, rs in groups.items():
            st = stats[k]
            L.append(
                f"| {label(k)} | "
                f"{mean_sd(pick(st, split, 'pred_share', INC))} | "
                f"{fmt(st[0][split]['true_share'][INC])} | "
                f"{mean_sd(pick(st, split, 'precision', INC))} | "
                f"{mean_sd(pick(st, split, 'recall', INC))} | "
                f"{mean_sd(pick(st, split, 'f1', INC))} | "
                f"{mean_sd(pick(st, split, 'macro_f1'))} |")
        L.append("")

    # 4. seen vs unseen
    L += ["## 4. Seen vs unseen claims", "",
          "On seen rows, always predicting NEI is correct at the NEI share "
          "shown, so accuracy there is not evidence of reading the abstract.",
          ""]
    for split in SPLITS:
        L += [f"### {split}", "",
              "| System | Seen n | Seen NEI share (= always-NEI acc) | "
              "Model acc, seen | Unseen n | Always-NEI acc, unseen | "
              "Model acc, unseen | Macro F1, unseen |",
              "|---|---|---|---|---|---|---|---|"]
        for k, rs in groups.items():
            st = stats[k]
            s0 = st[0][split]
            L.append(
                f"| {label(k)} | {s0['seen_n']} | "
                f"{fmt(s0['seen_nei_share'])} | "
                f"{mean_sd(pick(st, split, 'seen_acc'))} | "
                f"{s0['unseen_n']} | {fmt(s0['unseen_nei_share'])} | "
                f"{mean_sd(pick(st, split, 'unseen_acc'))} | "
                f"{mean_sd(pick(st, split, 'unseen_macro_f1'))} |")
        L.append("")
    L += ["Unseen-row F1 per class (validation, mean over seeds):", "",
          "| System | " + " | ".join(NAMES) + " |",
          "|---|" + "---|" * len(NAMES)]
    for k in groups:
        per = []
        for n in NAMES:
            vals = []
            for r in groups[k]:
                df = r["preds"]["validation"]
                un = df[df["claim_unseen"]]
                m = classification_metrics(un["label"], un["pred"])
                vals.append(m["per_class"][n]["f1"])
            per.append(mean_sd(vals))
        L.append(f"| {label(k)} | " + " | ".join(per) + " |")
    L.append("")

    # 5. limitation
    L += ["## 5. Limitation: repeated claims and label imbalance", "",
          f"- {rc['rows_in_repeated']} rows ({rc['repeated_claims']} claims) "
          "share their claim text with at least one other row. These rows "
          f"are {rc['nei_repeated']:.0%} NEI, versus "
          f"{rc['nei_unique']:.0%} for the {rc['rows_unique']} rows with a "
          "unique claim.",
          f"- Within train alone: {rc['nei_repeated_train']:.0%} NEI "
          f"(repeated) vs {rc['nei_unique_train']:.0%} (unique), so this is "
          "not an artifact of the split.",
          f"- In {rc['claims_one_nct']} of {rc['repeated_claims']} repeated "
          "claims all rows share one nctId, and "
          f"{rc['claims_no_non_nei']} of them have no non-NEI row. This is "
          "consistent with one trial claim being paired with several "
          "papers of which at most one reports the outcome. We checked this "
          "by counts and a single example, not abstract by abstract.",
          "- Seen validation/test rows pair a train claim with a different "
          "paper (no shared PMID or abstract), so rows are not duplicated, "
          "but the claim text carries a strong NEI prior.",
          "- All-rows scores are therefore inflated by an easy NEI "
          "majority among seen claims. Unseen-claim scores are the better "
          "estimate of verification skill, and both are reported.",
          "- Claim text is not the only cue a model could use; we did not "
          "test a claim-only model, so this remains a hypothesis.",
          "- For retrieval (Phase 2), the paired PMID is a gold document "
          "only for Evidence and Inconclusive rows; for NEI rows it is a "
          "distractor.", ""]

    # 6. reproducibility
    L += ["## 6. Runs included", ""]
    for k, rs in groups.items():
        for r in rs:
            c = r["config"]
            L.append(f"- {label(k)} seed {seed_of(r)}: lr "
                     f"{c['args']['lr']}, epochs {c['args']['epochs']}, "
                     f"batch {c['effective_batch_size']}, "
                     f"{c['device']}, transformers "
                     f"{c['packages'].get('transformers')}, "
                     f"`{r['dir'].name}`")
    L.append("")

    summary = {
        "best_group_by_val_macro_f1": f"{best[0]}|{best[1]}",
        "groups": {f"{k[0]}|{k[1]}": {
            "seeds": [seed_of(r) for r in rs],
            "stats": stats[k]} for k, rs in groups.items()},
        "head_tail_minus_head": wins,
        "majority": {"label": maj_name, **maj},
        "repeated_claims": rc,
    }
    return "\n".join(L), summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default=str(RESULTS_DIR))
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    dirs = discover(results_dir)
    if not dirs:
        raise SystemExit(f"No full E1 runs found in {results_dir}")
    groups: dict = defaultdict(list)
    for d in dirs:
        run = load_run(d)
        run["stats"] = run_stats(run)
        groups[run_key(run)].append(run)
    for rs in groups.values():
        rs.sort(key=seed_of)
    groups = dict(sorted(groups.items()))

    md, summary = build_report(groups, load_all())
    out = Path(args.out_dir) if args.out_dir else results_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "e1_report.md").write_text(md)
    (out / "e1_report.json").write_text(json.dumps(summary, indent=2))
    print(md)
    print(f"\nWrote {out / 'e1_report.md'}")


if __name__ == "__main__":
    main()
