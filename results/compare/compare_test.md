# System comparison (test, n=355)

| System | Macro F1 | 95% CI | Acc | Unseen macro F1 |
|---|---|---|---|---|
| B0 | 0.232 | 0.217-0.246 | 0.535 | 0.191 |
| B0-knowledge | 0.194 | 0.150-0.235 | 0.208 | 0.224 |
| E2 top-1 | 0.554 | 0.505-0.605 | 0.608 | 0.546 |
| E3 | 0.565 | 0.516-0.613 | 0.617 | 0.554 |
| B0+correction | 0.746 | 0.694-0.796 | 0.763 | 0.790 |

Paired difference in macro F1 (A - B):

| A vs B | diff | 95% CI | excludes 0 |
|---|---|---|---|
| B0+correction vs E2 top-1 | +0.192 | +0.133..+0.250 | yes |
| B0+correction vs E3 | +0.181 | +0.120..+0.241 | yes |
| B0+correction vs B0 | +0.513 | +0.456..+0.568 | yes |
| B0+correction vs B0-knowledge | +0.551 | +0.491..+0.613 | yes |
| E3 vs E2 top-1 | +0.011 | -0.012..+0.033 | no |
