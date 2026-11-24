"""Dataset loading and preparation for CliniFact.

Keeps the official splits and labels unchanged. Only exact duplicate rows
(same claim, PMID and nctId) are removed, within the combined data.
"""
from __future__ import annotations

import pandas as pd

from .config import DATA_DIR, LABELS, SPLITS


def load_split(split: str) -> pd.DataFrame:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    return pd.read_csv(DATA_DIR / f"{split}_set.csv")


def load_all(drop_duplicates: bool = True) -> dict[str, pd.DataFrame]:
    """Return {split: DataFrame} with cleaning and a `claim_unseen` flag.

    claim_unseen is True when the claim text does not occur in the train split
    (always False for train itself). Used to report results on a subset where
    claim-text memorization is impossible.
    """
    data = {s: load_split(s) for s in SPLITS}

    key = ["claim", "PMID", "nctId"]
    if drop_duplicates:
        seen: set[tuple] = set()
        for s in SPLITS:  # train first, so the train copy is the one kept
            df = data[s]
            '''
            keys = [("Aspirin reduces fever", "12345", "NCT001"),
                     ("Vitamin C supports immunity", "67890", "NCT002"),
                     ("Aspirin reduces fever", "12345", "NCT001")]
            '''
            keys = list(map(tuple, df[key].astype(str).values))
            keep = []
            for k in keys:
                keep.append(k not in seen)
                seen.add(k)
            data[s] = df[keep].reset_index(drop=True)

    train_claims = set(data["train"]["claim"])
    for s in SPLITS:
        data[s]["claim_unseen"] = (
            ~data[s]["claim"].isin(train_claims) if s != "train" else False
        )
        _validate(data[s])
    return data


def _validate(df: pd.DataFrame) -> None:
    bad = set(df["label"].unique()) - set(LABELS)
    if bad:
        raise ValueError(f"unexpected labels: {bad}")
    if df["claim"].isna().any() or df["article_abstract"].isna().any():
        raise ValueError("missing claim or abstract")


def to_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Model-ready pairs: only claim, abstract and label (plus ids for analysis).

    Label-derived columns (pValues, statisticalMethods, ...) are deliberately
    excluded.
    """
    out = df[["claim", "article_abstract", "label", "PMID", "nctId",
              "claim_unseen"]].copy()
    out["claim"] = out["claim"].str.strip()
    out["article_abstract"] = out["article_abstract"].str.strip()
    return out.reset_index(drop=True)
