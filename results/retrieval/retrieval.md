Dense weight (val-tuned): 0.3

| system | split | subset | n | R@5 | R@10 | MRR |
|---|---|---|---|---|---|---|
| bm25 | validation | all | 149 | 0.9161 | 0.9161 | 0.8151 |
| bm25 | validation | unseen | 142 | 0.912 | 0.912 | 0.8189 |
| bm25 | test | all | 165 | 0.9212 | 0.9424 | 0.8137 |
| bm25 | test | unseen | 154 | 0.9351 | 0.9545 | 0.8265 |
| medcpt | validation | all | 149 | 0.849 | 0.9027 | 0.7571 |
| medcpt | validation | unseen | 142 | 0.8415 | 0.8979 | 0.7533 |
| medcpt | test | all | 165 | 0.8697 | 0.9333 | 0.7379 |
| medcpt | test | unseen | 154 | 0.8896 | 0.9481 | 0.7609 |
| hybrid_rrf | validation | all | 149 | 0.9094 | 0.9195 | 0.8283 |
| hybrid_rrf | validation | unseen | 142 | 0.9049 | 0.9155 | 0.8234 |
| hybrid_rrf | test | all | 165 | 0.9333 | 0.9545 | 0.8385 |
| hybrid_rrf | test | unseen | 154 | 0.9481 | 0.9675 | 0.8525 |