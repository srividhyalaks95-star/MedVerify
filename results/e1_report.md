# E1 report: BioBERT claim + abstract (multi-seed)

Scope: CliniFact claim verification with pre-paired abstracts (not general medical QA). 'Unseen' = claim text not present in train; 'seen' = it is.

## 1. Headline (mean ± sd over seeds)

| System | Seeds | Val macro F1 | Val macro F1, unseen | Test macro F1 | Test macro F1, unseen | Test acc |
|---|---|---|---|---|---|---|
| Majority (NEI) | - | 0.230 | 0.151 | 0.244 | 0.199 | 0.576 |
| BioBERT `head` / weighted | 3 | 0.718 ± 0.025 | 0.689 ± 0.023 | 0.635 ± 0.032 | 0.614 ± 0.033 | 0.708 ± 0.027 |
| BioBERT `head_tail` / weighted | 3 | 0.743 ± 0.026 | 0.715 ± 0.023 | 0.666 ± 0.051 | 0.648 ± 0.053 | 0.730 ± 0.034 |

Best group by mean validation macro F1: **`head_tail` / weighted**. Test numbers were not used for selection.

## 2. head vs head_tail, seed by seed (head_tail minus head)

### weighted

| Seed | Δ val macro F1 | Δ val unseen | Δ test macro F1 | Δ test unseen |
|---|---|---|---|---|
| 42 | +0.046 | +0.053 | +0.059 | +0.072 |
| 43 | +0.030 | +0.025 | +0.047 | +0.047 |
| 44 | -0.001 | +0.000 | -0.012 | -0.017 |
| **wins** | 2/3 | 3/3 | 2/3 | 2/3 |

## 3. Effect of class weighting on Inconclusive

Compare the predicted share of Inconclusive with its true share. A predicted share well above the true share means the model over-predicts Inconclusive.

### validation

| System | Inconclusive predicted | true | precision | recall | F1 | Macro F1 |
|---|---|---|---|---|---|---|
| `head` / weighted | 0.190 ± 0.024 | 0.212 | 0.568 ± 0.088 | 0.502 ± 0.017 | 0.532 ± 0.048 | 0.718 ± 0.025 |
| `head_tail` / weighted | 0.214 ± 0.042 | 0.212 | 0.605 ± 0.089 | 0.602 ± 0.075 | 0.600 ± 0.057 | 0.743 ± 0.026 |

### test

| System | Inconclusive predicted | true | precision | recall | F1 | Macro F1 |
|---|---|---|---|---|---|---|
| `head` / weighted | 0.217 ± 0.032 | 0.137 | 0.342 ± 0.060 | 0.531 ± 0.028 | 0.414 ± 0.047 | 0.635 ± 0.032 |
| `head_tail` / weighted | 0.217 ± 0.024 | 0.137 | 0.384 ± 0.092 | 0.605 ± 0.123 | 0.469 ± 0.102 | 0.666 ± 0.051 |

## 4. Seen vs unseen claims

On seen rows, always predicting NEI is correct at the NEI share shown, so accuracy there is not evidence of reading the abstract.

### validation

| System | Seen n | Seen NEI share (= always-NEI acc) | Model acc, seen | Unseen n | Always-NEI acc, unseen | Model acc, unseen | Macro F1, unseen |
|---|---|---|---|---|---|---|---|
| `head` / weighted | 114 | 0.939 | 0.927 ± 0.005 | 202 | 0.292 | 0.693 ± 0.028 | 0.689 ± 0.023 |
| `head_tail` / weighted | 114 | 0.939 | 0.933 ± 0.013 | 202 | 0.292 | 0.715 ± 0.022 | 0.715 ± 0.023 |

### test

| System | Seen n | Seen NEI share (= always-NEI acc) | Model acc, seen | Unseen n | Always-NEI acc, unseen | Model acc, unseen | Macro F1, unseen |
|---|---|---|---|---|---|---|---|
| `head` / weighted | 124 | 0.903 | 0.855 ± 0.016 | 270 | 0.426 | 0.641 ± 0.034 | 0.614 ± 0.033 |
| `head_tail` / weighted | 124 | 0.903 | 0.858 ± 0.020 | 270 | 0.426 | 0.672 ± 0.044 | 0.648 ± 0.053 |

Unseen-row F1 per class (validation, mean over seeds):

| System | Inconclusive | Evidence | NEI |
|---|---|---|---|
| `head` / weighted | 0.549 ± 0.042 | 0.718 ± 0.038 | 0.801 ± 0.008 |
| `head_tail` / weighted | 0.616 ± 0.054 | 0.722 ± 0.020 | 0.807 ± 0.012 |

## 5. Limitation: repeated claims and label imbalance

- 805 rows (277 claims) share their claim text with at least one other row. These rows are 90% NEI, versus 26% for the 1164 rows with a unique claim.
- Within train alone: 90% NEI (repeated) vs 23% (unique), so this is not an artifact of the split.
- In 270 of 277 repeated claims all rows share one nctId, and 208 of them have no non-NEI row. This is consistent with one trial claim being paired with several papers of which at most one reports the outcome. We checked this by counts and a single example, not abstract by abstract.
- Seen validation/test rows pair a train claim with a different paper (no shared PMID or abstract), so rows are not duplicated, but the claim text carries a strong NEI prior.
- All-rows scores are therefore inflated by an easy NEI majority among seen claims. Unseen-claim scores are the better estimate of verification skill, and both are reported.
- Claim text is not the only cue a model could use; we did not test a claim-only model, so this remains a hypothesis.
- For retrieval (Phase 2), the paired PMID is a gold document only for Evidence and Inconclusive rows; for NEI rows it is a distractor.

## 6. Runs included

- `head` / weighted seed 42: lr 2e-05, epochs 5, batch 16, Tesla T4, transformers 5.22.1, `e1_biobert_head_seed42`
- `head` / weighted seed 43: lr 2e-05, epochs 5, batch 16, Tesla T4, transformers 5.22.1, `e1_biobert_head_seed43`
- `head` / weighted seed 44: lr 2e-05, epochs 5, batch 16, Tesla T4, transformers 5.22.1, `e1_biobert_head_seed44`
- `head_tail` / weighted seed 42: lr 2e-05, epochs 5, batch 16, Tesla T4, transformers 5.22.1, `e1_biobert_head_tail_seed42`
- `head_tail` / weighted seed 43: lr 2e-05, epochs 5, batch 16, Tesla T4, transformers 5.22.1, `e1_biobert_head_tail_seed43`
- `head_tail` / weighted seed 44: lr 2e-05, epochs 5, batch 16, Tesla T4, transformers 5.22.1, `e1_biobert_head_tail_seed44`
