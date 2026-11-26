## Rule: top1 (PRIMARY)

| run | split | subset | n | macro F1 | acc |
|---|---|---|---|---|---|
| e1_biobert_head_tail_seed42 | validation | all | 288 | 0.651 | 0.677 |
| e1_biobert_head_tail_seed42 | validation | unseen | 198 | 0.630 | 0.631 |
| e1_biobert_head_tail_seed42 | test | all | 355 | 0.554 | 0.608 |
| e1_biobert_head_tail_seed42 | test | unseen | 257 | 0.546 | 0.568 |

Mean +- sd over runs (macro F1):
- validation_all: 0.651 +- 0.000 (acc 0.677)
- validation_unseen: 0.630 +- 0.000 (acc 0.631)
- test_all: 0.554 +- 0.000 (acc 0.608)
- test_unseen: 0.546 +- 0.000 (acc 0.568)

Test confusion matrix (e1_biobert_head_tail_seed42; rows=true, cols=pred, order ['Inconclusive', 'Evidence', 'NEI']):
[[23, 23, 8], [32, 68, 11], [47, 18, 125]]

## Rule: top3_any_non_nei (experimental variant)

| run | split | subset | n | macro F1 | acc |
|---|---|---|---|---|---|
| e1_biobert_head_tail_seed42 | validation | all | 288 | 0.583 | 0.597 |
| e1_biobert_head_tail_seed42 | validation | unseen | 198 | 0.577 | 0.571 |
| e1_biobert_head_tail_seed42 | test | all | 355 | 0.520 | 0.535 |
| e1_biobert_head_tail_seed42 | test | unseen | 257 | 0.512 | 0.506 |

Mean +- sd over runs (macro F1):
- validation_all: 0.583 +- 0.000 (acc 0.597)
- validation_unseen: 0.577 +- 0.000 (acc 0.571)
- test_all: 0.520 +- 0.000 (acc 0.535)
- test_unseen: 0.512 +- 0.000 (acc 0.506)

Test confusion matrix (e1_biobert_head_tail_seed42; rows=true, cols=pred, order ['Inconclusive', 'Evidence', 'NEI']):
[[38, 13, 3], [52, 52, 7], [76, 14, 100]]

Always-NEI baseline:
- validation: macro F1 0.217, acc 0.483
- test: macro F1 0.232, acc 0.535