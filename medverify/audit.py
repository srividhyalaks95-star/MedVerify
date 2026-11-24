"""Step 1 data audit. Run from Final_Project/:  python -m medverify.audit"""
import json

import numpy as np

from .config import BIOBERT_CHECKPOINT, LABELS, MAX_LENGTH, RESULTS_DIR
from .data import load_all, to_pairs


def main() -> None:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BIOBERT_CHECKPOINT)
    data = load_all()
    report: dict = {}
    for split, df in data.items():
        p = to_pairs(df)
        enc = tok(list(p.claim), list(p.article_abstract), truncation=False)
        lens = np.array([len(x) for x in enc["input_ids"]])
        claim_lens = np.array([len(x) for x in tok(list(p.claim))["input_ids"]])
        report[split] = {
            "rows": len(p),
            "labels": {LABELS[k]: int(v) for k, v in
                       p.label.value_counts().sort_index().items()},
            "claim_unseen_rows": int(p.claim_unseen.sum()),
            "pair_tokens_mean": float(lens.mean()),
            "pair_tokens_p95": float(np.percentile(lens, 95)),
            "pair_tokens_max": int(lens.max()),
            "pct_over_max_length": float((lens > MAX_LENGTH).mean() * 100),
            "claim_tokens_max": int(claim_lens.max()),
        }
        print(split, json.dumps(report[split], indent=2))
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "data_audit.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
