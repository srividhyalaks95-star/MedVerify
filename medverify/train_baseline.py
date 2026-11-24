"""E1: BioBERT claim/abstract classifier.

Run from Final_Project/ (on Colab GPU):
    python -m medverify.train_baseline --policy head
    python -m medverify.train_baseline --policy head_tail
Quick sanity check (tiny subset, 1 epoch):
    python -m medverify.train_baseline --policy head --smoke

Model selection uses validation macro F1 only. Test is predicted once, after
the best checkpoint is restored, and is never used for any decision.
"""
from __future__ import annotations

import argparse
import inspect
import json
import platform
import sys
from importlib import metadata

import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments,
                          set_seed)

from .config import (BIOBERT_CHECKPOINT, LABELS, LEAKY_COLUMNS, MAX_LENGTH,
                     NUM_LABELS, RESULTS_DIR, SEED)
from .data import load_all, to_pairs
from .encoding import POLICIES, encode_pairs
from .metrics import classification_metrics

PACKAGES = ("torch", "transformers", "numpy", "pandas", "scikit-learn",
            "accelerate", "datasets")


class PairDataset(torch.utils.data.Dataset):
    def __init__(self, enc: dict, labels):
        self.enc, self.labels = enc, list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = int(self.labels[i])
        return item


class WeightedTrainer(Trainer):
    """Trainer with class-weighted cross-entropy."""

    def __init__(self, *args, class_weights: torch.Tensor, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss_fn = nn.CrossEntropyLoss(
            weight=self.class_weights.to(outputs.logits.device))
        loss = loss_fn(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


def package_versions() -> dict:
    out = {}
    for p in PACKAGES:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = None
    return out


def compute_metrics(eval_pred) -> dict:
    logits, y = eval_pred
    m = classification_metrics(y, np.argmax(logits, axis=-1))
    out = {"accuracy": m["accuracy"], "macro_f1": m["macro_f1"]}
    for name, v in m["per_class"].items():
        out[f"f1_{name}"] = v["f1"]
    return out


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def save_predictions(trainer, ds, pairs: pd.DataFrame, path) -> dict:
    logits = trainer.predict(ds).predictions
    probs = softmax(logits)
    pred = probs.argmax(axis=1)
    out = pairs[["PMID", "nctId", "label", "claim_unseen"]].copy()
    out["pred"] = pred
    for i, name in LABELS.items():
        out[f"prob_{name}"] = probs[:, i]
    out.to_csv(path, index=False)
    return classification_metrics(pairs["label"].values, pred)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=POLICIES, required=True)
    ap.add_argument("--model", default=BIOBERT_CHECKPOINT)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch_size", type=int, default=16,
                    help="effective batch size")
    ap.add_argument("--per_device_batch", type=int, default=8,
                    help="micro-batch; grad accumulation fills the rest")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny subset, 1 epoch, for a quick sanity check")
    args = ap.parse_args()

    if args.batch_size % args.per_device_batch:
        sys.exit("batch_size must be divisible by per_device_batch")
    accum = args.batch_size // args.per_device_batch
    if args.smoke:
        args.epochs = 1

    run_name = f"e1_biobert_{args.policy}_seed{args.seed}" + (
        "_smoke" if args.smoke else "")
    out_dir = RESULTS_DIR / run_name if args.output_dir is None else \
        __import__("pathlib").Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    data = load_all()
    pairs = {s: to_pairs(df) for s, df in data.items()}
    if args.smoke:
        pairs = {s: p.sample(min(len(p), 64), random_state=args.seed)
                 .reset_index(drop=True) for s, p in pairs.items()}

    tok = AutoTokenizer.from_pretrained(args.model)
    ds = {}
    for s, p in pairs.items():
        enc = encode_pairs(tok, p.claim, p.article_abstract, args.policy,
                           MAX_LENGTH)
        ds[s] = PairDataset(enc, p.label)

    # Class weights from TRAIN only: n / (k * count_c)
    counts = np.bincount(pairs["train"].label, minlength=NUM_LABELS)
    weights = len(pairs["train"]) / (NUM_LABELS * np.maximum(counts, 1))
    class_weights = torch.tensor(weights, dtype=torch.float)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=NUM_LABELS,
        id2label={i: n for i, n in LABELS.items()},
        label2id={n: i for i, n in LABELS.items()})
    # The checkpoint's own config has 2 labels, which triggers a harmless
    # warning; confirm the head and label maps are the 3-class ones.
    assert model.classifier.out_features == NUM_LABELS
    assert model.config.id2label == {i: n for i, n in LABELS.items()}

    targs = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.per_device_batch,
        per_device_eval_batch_size=args.per_device_batch * 2,
        gradient_accumulation_steps=accum,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=1,
        seed=args.seed,
        fp16=torch.cuda.is_available(),
        report_to="none",
        logging_strategy="epoch",
    )
    trainer_kwargs = dict(
        model=model, args=targs, train_dataset=ds["train"],
        eval_dataset=ds["validation"], compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tok),
        class_weights=class_weights)
    # `tokenizer` was renamed `processing_class` in newer transformers
    if "processing_class" in inspect.signature(Trainer.__init__).parameters:
        trainer_kwargs["processing_class"] = tok
    else:
        trainer_kwargs["tokenizer"] = tok
    trainer = WeightedTrainer(**trainer_kwargs)

    train_result = trainer.train()

    val_metrics = save_predictions(trainer, ds["validation"],
                                   pairs["validation"],
                                   out_dir / "val_predictions.csv")
    test_metrics = save_predictions(trainer, ds["test"], pairs["test"],
                                    out_dir / "test_predictions.csv")

    config = {
        "run_name": run_name,
        "args": vars(args),
        "effective_batch_size": args.per_device_batch * accum,
        "label_mapping": {str(k): v for k, v in LABELS.items()},
        "class_weights": {LABELS[i]: float(w)
                          for i, w in enumerate(weights)},
        "excluded_columns": list(LEAKY_COLUMNS),
        "max_length": MAX_LENGTH,
        "rows": {s: len(p) for s, p in pairs.items()},
        "packages": package_versions(),
        "python": platform.python_version(),
        "device": (torch.cuda.get_device_name(0)
                   if torch.cuda.is_available() else "cpu"),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    (out_dir / "metrics.json").write_text(json.dumps({
        "validation": val_metrics, "test": test_metrics,
        "log_history": trainer.state.log_history,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "train_runtime_sec": train_result.metrics.get("train_runtime"),
    }, indent=2))
    print(f"[{run_name}] val macro F1 = {val_metrics['macro_f1']:.4f} | "
          f"test macro F1 = {test_metrics['macro_f1']:.4f}")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
