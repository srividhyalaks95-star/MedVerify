"""Claim + abstract pair encoding with configurable truncation.

Policies (only the abstract is ever truncated; the claim is kept whole):
  "head"      (Option A, baseline): keep the first tokens of the abstract.
  "head_tail" (Option B): keep the first and last tokens, drop the middle,
              so the conclusion is retained.
"""
from __future__ import annotations

from typing import Sequence

from .config import MAX_LENGTH

POLICIES = ("head", "head_tail")
MAX_CLAIM_TOKENS = 200  # safety cap; the longest claim in the data is 151


def _truncate(ids: list[int], budget: int, policy: str,
              head_fraction: float) -> list[int]:
    if len(ids) <= budget:
        return ids
    if policy == "head":
        return ids[:budget]
    head = int(round(budget * head_fraction))
    tail = budget - head
    return ids[:head] + (ids[-tail:] if tail > 0 else [])


def encode_pairs(tokenizer, claims: Sequence[str], abstracts: Sequence[str],
                 policy: str = "head", max_length: int = MAX_LENGTH,
                 head_fraction: float = 0.5) -> dict[str, list[list[int]]]:
    """Return input_ids, token_type_ids, attention_mask (unpadded lists)."""
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}")
    claim_ids = tokenizer(list(claims), add_special_tokens=False)["input_ids"]
    abs_ids = tokenizer(list(abstracts), add_special_tokens=False)["input_ids"]

    out: dict[str, list[list[int]]] = {
        "input_ids": [], "token_type_ids": [], "attention_mask": []}
    cls, sep = tokenizer.cls_token_id, tokenizer.sep_token_id
    n_special = 3  # [CLS] claim [SEP] abstract [SEP]
    for c, a in zip(claim_ids, abs_ids):
        c = c[:MAX_CLAIM_TOKENS]
        budget = max_length - n_special - len(c)
        a = _truncate(a, budget, policy, head_fraction)
        ids = [cls] + c + [sep] + a + [sep]
        out["input_ids"].append(ids)
        out["token_type_ids"].append([0] * (len(c) + 2) + [1] * (len(a) + 1))
        out["attention_mask"].append([1] * len(ids))
    return out
