# Colab setup (paste each block into a Colab cell)

Runtime > Change runtime type > GPU (T4 is enough for BioBERT-base).

```python
# 1. GPU check
import torch
assert torch.cuda.is_available(), "Switch the Colab runtime to GPU"
print(torch.cuda.get_device_name(0))
```

```bash
# 2. Get the code (Final_Project is the repo root). Replace the URL.
# Private repo: https://<token>@github.com/<you>/medverify.git
!git clone https://github.com/<you>/medverify.git
%cd medverify
```

```bash
# 3. CliniFact data is not part of this repo; clone it into external/
!git clone https://github.com/ds4dh/CliniFact.git external/CliniFact
```

```bash
# 4. Install dependencies (torch is preinstalled on Colab; don't reinstall it)
!pip install -q transformers accelerate scikit-learn pandas datasets
```

```bash
# 5. Smoke test (about a minute), then the two full runs
!python -m medverify.train_baseline --policy head --smoke
!python -m medverify.train_baseline --policy head
!python -m medverify.train_baseline --policy head_tail
```

```bash
%%bash
# 5b. Seeds 42-44, weighted and unweighted, both policies (12 runs).
# Existing run folders are skipped, so this can be re-run after a disconnect.
for s in 42 43 44; do
  for p in head head_tail; do
    [ -d results/e1_biobert_${p}_seed${s} ] || \
      python -m medverify.train_baseline --policy $p --seed $s
    [ -d results/e1_biobert_${p}_seed${s}_noweights ] || \
      python -m medverify.train_baseline --policy $p --seed $s --no_class_weights
  done
done
```

```python
# 6. Save results (checkpoints not needed) before the runtime resets
from google.colab import drive
drive.mount('/content/drive')
!cp -r results /content/drive/MyDrive/medverify_results
```
