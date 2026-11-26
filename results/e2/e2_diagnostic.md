Run: e1_biobert_head_tail_seed42

## validation
- claims with a gold PMID: 149; gold in top-3: 132, missed: 17
- accuracy when gold retrieved: 0.614
- accuracy when gold missed: 0.4117647058823529
- NEI claims: 139; predicted non-NEI: 55
- share of single abstracts of NEI claims voted non-NEI: 0.216
- aggregation variants (macro F1 / acc):
  - top3_any_non_nei (variant): 0.583 / 0.597
  - mean_prob: 0.544 / 0.611
  - top1_only (primary): 0.651 / 0.677

## test
- claims with a gold PMID: 165; gold in top-3: 150, missed: 15
- accuracy when gold retrieved: 0.567
- accuracy when gold missed: 0.3333333333333333
- NEI claims: 190; predicted non-NEI: 90
- share of single abstracts of NEI claims voted non-NEI: 0.289
- aggregation variants (macro F1 / acc):
  - top3_any_non_nei (variant): 0.520 / 0.535
  - mean_prob: 0.481 / 0.552
  - top1_only (primary): 0.554 / 0.608
