# System comparison (validation, n=288)

| System | Macro F1 | 95% CI | Acc | Unseen macro F1 |
|---|---|---|---|---|
| B0 | 0.217 | 0.199-0.234 | 0.483 | 0.147 |
| B0-knowledge | 0.216 | 0.169-0.265 | 0.271 | 0.250 |
| E2 top-1 | 0.651 | 0.593-0.708 | 0.677 | 0.630 |
| E3 | 0.657 | 0.600-0.714 | 0.681 | 0.644 |
| B0+correction | 0.686 | 0.627-0.737 | 0.698 | 0.681 |

Paired difference in macro F1 (A - B):

| A vs B | diff | 95% CI | excludes 0 |
|---|---|---|---|
| B0+correction vs E2 top-1 | +0.035 | -0.042..+0.109 | no |
| B0+correction vs E3 | +0.029 | -0.049..+0.104 | no |
| B0+correction vs B0 | +0.469 | +0.407..+0.526 | yes |
| B0+correction vs B0-knowledge | +0.470 | +0.400..+0.540 | yes |
| E3 vs E2 top-1 | +0.006 | -0.016..+0.029 | no |
