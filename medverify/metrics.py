"""Shared classification metrics (reused by baseline and later systems)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)

from .config import LABELS


def classification_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    ids = sorted(LABELS)
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=ids, zero_division=0)
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=ids,
                                   average="macro", zero_division=0)),
        "per_class": {
            LABELS[i]: {"precision": float(p[k]), "recall": float(r[k]),
                        "f1": float(f[k]), "support": int(s[k])}
            for k, i in enumerate(ids)
        },
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=ids).tolist(),
    }
