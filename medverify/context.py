"""E3 step 1-2: parse the claim template and score context alignment.

Claim template: "<A> is <relation> to <B> in terms of <OUTCOME>".
Population is not present in the template, so it is not used.
Alignment uses only claim and abstract text (no labels, no p-values).
"""
from __future__ import annotations

import re

_TEMPLATE = re.compile(r"^(.+?) is (.+?) to (.+?) in terms of (.+)$",
                       re.DOTALL)
_STOP = set("""a an the of in on at to for and or with without by from as is are
was were be been vs versus than that this these those it its into after before
during between among within over under per not no""".split())
# arm-name filler that carries no content
_GENERIC = set("""group groups arm arms treatment treated cohort control
comparison intervention experimental standard usual care participants
patients subjects number change baseline""".split())
_PLACEBO = {"placebo", "control", "sham", "usual", "standard", "vehicle"}
_TIME = re.compile(r"\b\d+(\.\d+)?\s*(hour|hours|day|days|week|weeks|month|"
                   r"months|year|years|wk|wks|hr|hrs)\b", re.I)


def stem(w: str) -> str:
    for suf in ("ations", "ation", "ings", "ing", "ies", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            return w[: -len(suf)]
    return w


def tokens(text: str, drop_generic: bool = True) -> list[str]:
    out = []
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        if w in _STOP or (drop_generic and w in _GENERIC):
            continue
        if len(w) < 2 and not w.isdigit():
            continue
        out.append(stem(w))
    return out


def parse_claim(claim: str) -> dict | None:
    m = _TEMPLATE.match(claim.strip())
    if not m:
        return None
    a, rel, b, outcome = (x.strip() for x in m.groups())
    outcome = _TIME.sub(" ", outcome)  # time frames are noisy for matching
    return {"intervention": a, "relation": rel, "comparator": b,
            "outcome": outcome,
            "int_tok": set(tokens(a)), "cmp_tok": set(tokens(b)),
            "out_tok": set(tokens(outcome))}


def _coverage(needed: set[str], abstract_tok: set[str]) -> float | None:
    if not needed:
        return None
    return len(needed & abstract_tok) / len(needed)


def alignment(parsed: dict, abstract: str) -> dict:
    """Coverage in [0,1] of each claim part by the abstract (None if the
    part has no content tokens after cleaning)."""
    at = set(tokens(abstract, drop_generic=False))
    cmp_cov = _coverage(parsed["cmp_tok"], at)
    cmp_is_placebo = bool(parsed["cmp_tok"] & _PLACEBO) or \
        any(w in _PLACEBO for w in re.findall(r"[a-z]+",
                                              parsed["comparator"].lower()))
    if cmp_is_placebo and cmp_cov is not None:
        # placebo/control arms are rarely named reliably; do not penalise
        cmp_cov = max(cmp_cov, 1.0 if (at & _PLACEBO) else cmp_cov)
    return {"int_cov": _coverage(parsed["int_tok"], at),
            "cmp_cov": cmp_cov,
            "out_cov": _coverage(parsed["out_tok"], at)}
