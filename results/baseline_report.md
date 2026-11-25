# E1 baseline report (BioBERT claim + abstract)

Scope: CliniFact clinical-research claim verification with pre-paired abstracts. Not general medical QA.

Labels: 0=Inconclusive, 1=Evidence, 2=NEI

## 1. Headline: macro F1 and accuracy

| System | Val macro F1 | Val acc | Test macro F1 | Test acc |
|---|---|---|---|---|
| Majority class (NEI) | 0.230 | 0.525 | 0.244 | 0.576 |
| BioBERT `head` | 0.691 | 0.756 | 0.609 | 0.688 |
| BioBERT `head_tail` (selected) | 0.738 | 0.788 | 0.667 | 0.726 |

Selected policy: **`head_tail`**, chosen on validation macro F1 only. Test numbers were not used for selection.

## 2. Truncation comparison (validation)

| Policy | Macro F1 | F1 Inconclusive | F1 Evidence | F1 NEI | Best epoch |
|---|---|---|---|---|---|
| `head` | 0.691 | 0.489 | 0.674 | 0.911 | 5.0 |
| `head_tail` | 0.738 | 0.600 | 0.692 | 0.920 | 5.0 |

Absolute validation macro-F1 gap: 0.046. With 316 validation rows and a single seed, treat this difference as tentative until Step 12 adds bootstrap intervals and extra seeds.

## 3. Run detail: `head`

### validation: all rows (n=316)

Accuracy 0.756, macro F1 0.691

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Inconclusive | 0.485 | 0.493 | 0.489 | 67 |
| Evidence | 0.652 | 0.699 | 0.674 | 83 |
| NEI | 0.931 | 0.892 | 0.911 | 166 |

Confusion matrix:

| true \ pred | Inconclusive | Evidence | NEI |
|---|---|---|---|
| **Inconclusive** | 33 | 28 | 6 |
| **Evidence** | 20 | 58 | 5 |
| **NEI** | 15 | 3 | 148 |

### validation: claim-unseen subset (n=202)

Accuracy 0.663, macro F1 0.666

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Inconclusive | 0.525 | 0.500 | 0.512 | 64 |
| Evidence | 0.655 | 0.696 | 0.675 | 79 |
| NEI | 0.825 | 0.797 | 0.810 | 59 |

### test: all rows (n=394)

Accuracy 0.688, macro F1 0.609

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Inconclusive | 0.293 | 0.500 | 0.370 | 54 |
| Evidence | 0.639 | 0.611 | 0.624 | 113 |
| NEI | 0.902 | 0.771 | 0.831 | 227 |

Confusion matrix:

| true \ pred | Inconclusive | Evidence | NEI |
|---|---|---|---|
| **Inconclusive** | 27 | 21 | 6 |
| **Evidence** | 31 | 69 | 13 |
| **NEI** | 34 | 18 | 175 |

### test: claim-unseen subset (n=270)

Accuracy 0.611, macro F1 0.583

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Inconclusive | 0.321 | 0.481 | 0.385 | 52 |
| Evidence | 0.647 | 0.641 | 0.644 | 103 |
| NEI | 0.822 | 0.643 | 0.722 | 115 |

## 3. Run detail: `head_tail`

### validation: all rows (n=316)

Accuracy 0.788, macro F1 0.738

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Inconclusive | 0.542 | 0.672 | 0.600 | 67 |
| Evidence | 0.740 | 0.651 | 0.692 | 83 |
| NEI | 0.938 | 0.904 | 0.920 | 166 |

Confusion matrix:

| true \ pred | Inconclusive | Evidence | NEI |
|---|---|---|---|
| **Inconclusive** | 45 | 17 | 5 |
| **Evidence** | 24 | 54 | 5 |
| **NEI** | 14 | 2 | 150 |

### validation: claim-unseen subset (n=202)

Accuracy 0.713, macro F1 0.718

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Inconclusive | 0.581 | 0.672 | 0.623 | 64 |
| Evidence | 0.757 | 0.671 | 0.711 | 79 |
| NEI | 0.828 | 0.814 | 0.821 | 59 |

### test: all rows (n=394)

Accuracy 0.726, macro F1 0.667

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Inconclusive | 0.375 | 0.667 | 0.480 | 54 |
| Evidence | 0.733 | 0.655 | 0.692 | 113 |
| NEI | 0.893 | 0.775 | 0.830 | 227 |

Confusion matrix:

| true \ pred | Inconclusive | Evidence | NEI |
|---|---|---|---|
| **Inconclusive** | 36 | 11 | 7 |
| **Evidence** | 25 | 74 | 14 |
| **NEI** | 35 | 16 | 176 |

### test: claim-unseen subset (n=270)

Accuracy 0.674, macro F1 0.655

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Inconclusive | 0.425 | 0.654 | 0.515 | 52 |
| Evidence | 0.747 | 0.689 | 0.717 | 103 |
| NEI | 0.811 | 0.670 | 0.733 | 115 |

## 4. Memorization check: all rows vs claim-unseen (test)

| Policy | Macro F1 (all) | Macro F1 (claim-unseen) | Gap |
|---|---|---|---|
| `head` | 0.609 | 0.583 | 0.025 |
| `head_tail` | 0.667 | 0.655 | 0.012 |

A large positive gap would suggest the model benefits from claim text seen in training.

## 5. Reproducibility

- `head`: seed 42, lr 2e-05, epochs 5, effective batch 16, device Tesla T4, transformers 5.22.1, torch 2.14.1+cu130
- `head_tail`: seed 42, lr 2e-05, epochs 5, effective batch 16, device Tesla T4, transformers 5.22.1, torch 2.14.1+cu130
