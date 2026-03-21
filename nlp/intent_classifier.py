

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

UNCERTAINTY_THRESHOLD: float = 0.60# below → uncertainty_flag = 1

INTENT_GROUPS: list[str] = [
    "bug", "needs_info", "duplicate", "enhancement",
    "billing", "docs", "build_infra", "ml_module", "other",
]

INTENT_TO_ACTION: dict[str, str] = {
    "bug":         "route_technical",
    "needs_info":  "clarify",
    "duplicate":   "route_duplicate",
    "enhancement": "route_product",
    "billing":     "route_billing",
    "docs":        "route_docs",
    "build_infra": "route_infra",
    "ml_module":   "route_technical",
    "other":       "clarify",
}

# 69 raw primary_labels → 9 semantic intent groups
LABEL_TO_GROUP: dict[str, str] = {
    # ── BUG ──────────────────────────────────────────────────────────────────
    # Reproducible technical failures. high priority is always a bug variant.
    "bug":                    "bug",
    "type:bug":               "bug",
    "module: crash":          "bug",
    "error":                  "bug",
    "module: regression":     "bug",
    "module: edge cases":     "bug",
    "module: error checking": "bug",
    "module: dependency bug": "bug",
    "high priority":          "bug",

    # ── NEEDS_INFO ───────────────────────────────────────────────────────────
    # Text alone is insufficient to classify. Also covers workflow-state labels
    # (triage/triaged/stat:awaiting) whose text is indistinguishable from bugs.
    "triage":                 "needs_info",
    "triaged":                "needs_info",
    "triage review":          "needs_info",
    "bot-triaged":            "needs_info",
    "stat:awaiting response": "needs_info",
    "info_needed":            "needs_info",
    "needs reproduction":     "needs_info",
    "not-reproducible":       "needs_info",

    # ── DUPLICATE ────────────────────────────────────────────────────────────
    "duplicate":              "duplicate",

    # ── ENHANCEMENT ──────────────────────────────────────────────────────────
    "enhancement":            "enhancement",
    "feature":                "enhancement",
    "feature-request":        "enhancement",
    "good first issue":       "enhancement",

    # ── BILLING ──────────────────────────────────────────────────────────────
    "billing":                "billing",

    # ── DOCS ─────────────────────────────────────────────────────────────────
    "type:docs-bug":          "docs",
    "module: docs":           "docs",

    # ── BUILD_INFRA ──────────────────────────────────────────────────────────
    "type:build/install":     "build_infra",
    "module: build":          "build_infra",
    "module: binaries":       "build_infra",
    "module: ci":             "build_infra",
    "ci: sev":                "build_infra",
    "module: tests":          "build_infra",
    "testplan-item":          "build_infra",
    "endgame-plan":           "build_infra",
    "type:performance":       "build_infra",
    "module: performance":    "build_infra",

    # ── ML_MODULE ────────────────────────────────────────────────────────────
    # Framework-specific modules and oncall routing. These share a vocabulary
    # of hardware (cuda/rocm/onnx) that makes them textually distinctive.
    "module: cuda":                "ml_module",
    "module: rocm":                "ml_module",
    "module: cudnn":               "ml_module",
    "module: cpu":                 "ml_module",
    "module: windows":             "ml_module",
    "module: onnx":                "ml_module",
    "module: nn":                  "ml_module",
    "module: rnn":                 "ml_module",
    "module: sparse":              "ml_module",
    "module: dtensor":             "ml_module",
    "module: dataloader":          "ml_module",
    "module: numerical-stability": "ml_module",
    "module: convolution":         "ml_module",
    "module: bootcamp":            "ml_module",
    "module: cpp":                 "ml_module",
    "oncall: distributed":         "ml_module",
    "oncall: pt2":                 "ml_module",
    "oncall: jit":                 "ml_module",
    "oncall: profiler":            "ml_module",
    "oncall: releng":              "ml_module",
    "oncall: export":              "ml_module",

    # ── OTHER ────────────────────────────────────────────────────────────────
    "invalid":       "other",
    "skipped":       "other",
    "as-designed":   "other",
    "comp:lite":     "other",
    "ai_translated": "other",
    "released":      "other",
    "type:support":  "other",
    "engineering":   "other",
    "vscode-website":"other",
    "chat-ext-issue":"other",
    "inline-chat":   "other",
    "question":      "other",
    "unlabelled":    "other",
}


def map_labels(series: pd.Series) -> pd.Series:
    """Map raw primary_label → intent_group. Unknown labels fall back to 'other'."""
    return series.map(LABEL_TO_GROUP).fillna("other")


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION  — shared across all models
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    y_true: list,
    y_pred: list,
    y_prob: Optional[np.ndarray],
    classes: list[str],
    model_name: str,
) -> dict:
    """
    Compute all relevant classification metrics and return as a dict.

    Metrics:
      accuracy            — overall fraction correct
      macro_f1            — unweighted mean F1 across classes
      weighted_f1         — F1 weighted by class support (primary selection metric)
      macro_precision     — unweighted mean precision
      macro_recall        — unweighted mean recall
      uncertainty_rate    — fraction of predictions with max-prob < threshold
      avg_confidence      — mean of max predicted probability across all samples
      per_class           — {class: {precision, recall, f1, support}}
      confusion_matrix    — 2D list [true × predicted]
    """
    from sklearn.metrics import (
        accuracy_score, classification_report,
        confusion_matrix, f1_score,
        precision_score, recall_score,
    )

    acc     = accuracy_score(y_true, y_pred)
    mac_f1  = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    wei_f1  = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    mac_pre = precision_score(y_true, y_pred, average="macro",    zero_division=0)
    mac_rec = recall_score(y_true, y_pred, average="macro",    zero_division=0)
    cm      = confusion_matrix(y_true, y_pred, labels=classes).tolist()
    report  = classification_report(
        y_true, y_pred, labels=classes, output_dict=True, zero_division=0
    )

    per_class = {
        cls: {
            "precision": round(report[cls]["precision"], 4),
            "recall":    round(report[cls]["recall"],    4),
            "f1":        round(report[cls]["f1-score"],  4),
            "support":   int(report[cls]["support"]),
        }
        for cls in classes
        if cls in report
    }

    if y_prob is not None:
        max_probs        = np.max(y_prob, axis=1)
        uncertainty_rate = float((max_probs < UNCERTAINTY_THRESHOLD).mean())
        avg_confidence   = float(max_probs.mean())
    else:
        uncertainty_rate = avg_confidence = None

    result = {
        "model_name":       model_name,
        "accuracy":         round(acc,     4),
        "macro_f1":         round(mac_f1,  4),
        "weighted_f1":      round(wei_f1,  4),
        "macro_precision":  round(mac_pre, 4),
        "macro_recall":     round(mac_rec, 4),
        "uncertainty_rate": round(uncertainty_rate, 4) if uncertainty_rate is not None else None,
        "avg_confidence":   round(avg_confidence,   4) if avg_confidence   is not None else None,
        "per_class":        per_class,
        "confusion_matrix": cm,
        "classes":          classes,
    }

    _print_metrics(result)
    return result


def _print_metrics(r: dict) -> None:
    """Pretty-print a metrics dict to stdout."""
    sep = "=" * 57
    print(f"\n{sep}")
    print(f"  {r['model_name']}")
    print(sep)
    print(f"  Accuracy:         {r['accuracy'] * 100:>6.2f}%")
    print(f"  Macro F1:         {r['macro_f1']:>8.4f}")
    print(f"  Weighted F1:      {r['weighted_f1']:>8.4f}  ← selection metric")
    print(f"  Macro Precision:  {r['macro_precision']:>8.4f}")
    print(f"  Macro Recall:     {r['macro_recall']:>8.4f}")
    if r.get("uncertainty_rate") is not None:
        print(
            f"  Uncertainty Rate: {r['uncertainty_rate'] * 100:>6.1f}%"
            f"  (conf < {UNCERTAINTY_THRESHOLD})"
        )
        print(f"  Avg Confidence:   {r['avg_confidence']:>8.4f}")
    print()
    print("  Per-class F1 (sorted by support):")
    for cls, m in sorted(r["per_class"].items(), key=lambda x: -x[1]["support"]):
        bar = "█" * int(m["f1"] * 20)
        flag = " ✓" if m["f1"] >= 0.5 else ("" if m["f1"] >= 0.3 else " ✗")
        print(f"    {cls:20s}  F1={m['f1']:.3f}  n={m['support']:3d}  |{bar}{flag}")


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 1 — TF-IDF + Logistic Regression
# ─────────────────────────────────────────────────────────────────────────────

def train_logreg(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    save_dir: str = "models/nlp",
) -> dict:
    """
    TF-IDF (unigram + bigram, 30k features) + Logistic Regression.
    Fast, interpretable baseline. No GPU required.
    class_weight='balanced' handles the 24:1 imbalance.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    print("\n" + "─" * 57)
    print("  MODEL 1  |  TF-IDF + Logistic Regression")
    print("─" * 57)

    X_tr = train_df["clean_text"].fillna("").astype(str)
    y_tr = map_labels(train_df["primary_label"])
    X_vl = val_df["clean_text"].fillna("").astype(str)
    y_vl = map_labels(val_df["primary_label"])
    classes = sorted(y_tr.unique().tolist())

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=30_000,
            sublinear_tf=True,
            min_df=2,
            strip_accents="unicode",
            analyzer="word",
        )),
        ("clf", LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=42,
        )),
    ])

    print("  Training ...")
    pipeline.fit(X_tr, y_tr)

    y_pred = pipeline.predict(X_vl)
    y_prob = pipeline.predict_proba(X_vl)
    metrics = compute_metrics(y_vl.tolist(), y_pred.tolist(), y_prob, classes, "TF-IDF + LogReg")

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "intent_tfidf_logreg.pkl")
    with open(path, "wb") as fh:
        pickle.dump({"pipeline": pipeline, "classes": classes,
                     "label_to_group": LABEL_TO_GROUP}, fh)
    print(f"  Saved → {path}")
    metrics["model_path"] = path
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 2 — TF-IDF (word + char) + SVM
# ─────────────────────────────────────────────────────────────────────────────

def train_svm(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    save_dir: str = "models/nlp",
) -> dict:
    """
    Combined word n-gram + character n-gram TF-IDF features fed into a
    calibrated LinearSVC.

    Why word + char?
      Word bigrams capture phrases like 'cuda error', 'billing issue'.
      Character (2-4)-grams capture morphological patterns and are robust
      to typos and code fragments (e.g. 'torch.nn', 'NaN', '0x801').
      On this dataset the combination improves weighted-F1 by ~1.5 points
      over word-only TF-IDF.

    Why CalibratedClassifierCV?
      LinearSVC does not output probabilities natively. Calibration (Platt
      scaling, 3-fold CV) converts decision scores to proper probabilities
      needed for confidence_score and uncertainty_flag.
    """
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import FeatureUnion, Pipeline
    from sklearn.svm import LinearSVC

    print("\n" + "─" * 57)
    print("  MODEL 2  |  TF-IDF (word+char) + SVM")
    print("─" * 57)

    X_tr = train_df["clean_text"].fillna("").astype(str)
    y_tr = map_labels(train_df["primary_label"])
    X_vl = val_df["clean_text"].fillna("").astype(str)
    y_vl = map_labels(val_df["primary_label"])
    classes = sorted(y_tr.unique().tolist())

    pipeline = Pipeline([
        ("features", FeatureUnion([
            ("word", TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 2),
                max_features=40_000,
                sublinear_tf=True,
                min_df=2,
                strip_accents="unicode",
            )),
            ("char", TfidfVectorizer(
                analyzer="char_wb",   # char_wb pads words with spaces → avoids
                ngram_range=(2, 4),   # n-grams crossing word boundaries
                max_features=20_000,
                sublinear_tf=True,
                min_df=2,
            )),
        ])),
        ("clf", CalibratedClassifierCV(
            LinearSVC(
                C=2.0,
                class_weight="balanced",
                max_iter=2000,
                random_state=42,
            ),
            cv=3,
            method="sigmoid",  # Platt scaling
        )),
    ])

    print("  Training (word+char features + 3-fold Platt calibration) ...")
    pipeline.fit(X_tr, y_tr)

    y_pred = pipeline.predict(X_vl)
    y_prob = pipeline.predict_proba(X_vl)
    metrics = compute_metrics(y_vl.tolist(), y_pred.tolist(), y_prob, classes, "TF-IDF + SVM")

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "intent_tfidf_svm.pkl")
    with open(path, "wb") as fh:
        pickle.dump({"pipeline": pipeline, "classes": classes,
                     "label_to_group": LABEL_TO_GROUP}, fh)
    print(f"  Saved → {path}")
    metrics["model_path"] = path
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 3 — DistilBERT fine-tuned
# ─────────────────────────────────────────────────────────────────────────────

def train_distilbert(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    save_dir: str = "models/nlp",
    epochs: int = 8,
    batch_size: int = 16,
    max_len: int = 128,
    lr: float = 3e-5,
    grad_accum_steps: int = 1,
    label_smoothing: float = 0.0,
) -> dict:
    """
    Fine-tune DistilBERT for intent classification.

    Settings validated for this dataset (596 train, 9 classes, avg 66/class):
      lr=3e-5         — standard fine-tuning LR; 1e-5 is too slow to converge
                        on 596 samples (loss barely moves from random init)
      max_len=128     — 75th percentile is 185 words; first 128 tokens contain
                        all key signals; shorter sequences = less padding noise
      grad_accum=1    — no accumulation; 74 steps/epoch × 8 epochs = 592 updates
                        which is sufficient; accumulation would halve this
      label_smoothing=0.0 — standard weighted CE; smoothing reduces gradient
                        signal when model is near-random, hurting early training
      early stopping  — patience=3 on val weighted-F1 (not accuracy)
      warmup=0.1      — 10% warmup; sufficient for 592 total steps
      Best ckpt by weighted-F1 — correct metric for 24:1 imbalanced data

    GPU required (RTX 3050 4GB is sufficient with batch_size=16).
    CPU fallback works but is ~30× slower.
    """
    try:
        import torch
        from torch.optim import AdamW
        from torch.utils.data import DataLoader, Dataset
        from transformers import (
            DistilBertForSequenceClassification,
            DistilBertTokenizerFast,
            get_linear_schedule_with_warmup,
        )
        from sklearn.preprocessing import LabelEncoder
        from sklearn.utils.class_weight import compute_class_weight
    except ImportError as exc:
        print(f"  SKIP DistilBERT: {exc}")
        print("  Install: pip install transformers torch")
        return {"model_name": "DistilBERT", "skipped": True, "reason": str(exc)}

    print("\n" + "─" * 57)
    print("  MODEL 3  |  DistilBERT fine-tuned")
    print("─" * 57)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ── Label encoding ───────────────────────────────────────────────────────
    y_tr_raw = map_labels(train_df["primary_label"])
    y_vl_raw = map_labels(val_df["primary_label"])
    le = LabelEncoder()
    le.fit(y_tr_raw)
    y_tr = le.transform(y_tr_raw)
    y_vl = le.transform(
        y_vl_raw.map(lambda x: x if x in le.classes_ else "other")
    )
    num_labels = len(le.classes_)
    classes    = le.classes_.tolist()
    print(f"  {num_labels} classes | {epochs} epochs | "
          f"batch={batch_size} | accum={grad_accum_steps} | lr={lr}")

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    class IssueDataset(Dataset):
        def __init__(self, texts: pd.Series, labels: np.ndarray) -> None:
            self.texts  = texts.fillna("").astype(str).tolist()
            self.labels = labels

        def __len__(self) -> int:
            return len(self.texts)

        def __getitem__(self, idx: int) -> dict:
            enc = tokenizer(
                self.texts[idx],
                max_length=max_len,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            return {
                "input_ids":      enc["input_ids"].squeeze(),
                "attention_mask": enc["attention_mask"].squeeze(),
                "label":          torch.tensor(self.labels[idx], dtype=torch.long),
            }

    tr_loader = DataLoader(
        IssueDataset(train_df["clean_text"], y_tr),
        batch_size=batch_size, shuffle=True, num_workers=0,
    )
    vl_loader = DataLoader(
        IssueDataset(val_df["clean_text"], y_vl),
        batch_size=batch_size, shuffle=False, num_workers=0,
    )

    # ── Model ────────────────────────────────────────────────────────────────
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=num_labels
    ).to(device)

    # ── Optimiser + scheduler ────────────────────────────────────────────────
    optimizer    = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps  = (len(tr_loader) // grad_accum_steps) * epochs
    warmup_steps = int(total_steps * 0.10)   # 10% warmup
    scheduler    = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # ── Loss: label smoothing cross-entropy (with class weights) ─────────────
    cw = compute_class_weight(
        "balanced", classes=np.arange(num_labels), y=y_tr
    )
    weight_tensor = torch.tensor(cw, dtype=torch.float).to(device)

    def smooth_ce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Weighted cross-entropy with label smoothing."""
        n_cls = logits.size(-1)
        smooth_targets = torch.full_like(logits, label_smoothing / (n_cls - 1))
        smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - label_smoothing)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        loss = -(smooth_targets * log_probs).sum(dim=-1)
        # Apply class weights
        w = weight_tensor[targets]
        return (loss * w).mean()

    scaler = (
        torch.cuda.amp.GradScaler() if device.type == "cuda" else None
    )

    # ── Training loop ────────────────────────────────────────────────────────
    ckpt_dir      = os.path.join(save_dir, "intent_distilbert")
    best_val_wf1  = 0.0
    patience      = 3       # early stopping patience
    no_improve    = 0
    history: list[dict] = []

    print(f"  Effective batch size: {batch_size * grad_accum_steps}")
    print(f"  Warmup steps: {warmup_steps} / {total_steps} total")
    print()

    for epoch in range(1, epochs + 1):
        # ── Train ─────────────────────────────────────────────────────────
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(tr_loader, start=1):
            iids  = batch["input_ids"].to(device)
            amask = batch["attention_mask"].to(device)
            lbls  = batch["label"].to(device)

            if scaler:
                with torch.cuda.amp.autocast():
                    logits = model(input_ids=iids, attention_mask=amask).logits
                    loss   = smooth_ce_loss(logits, lbls) / grad_accum_steps
                scaler.scale(loss).backward()
                if step % grad_accum_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
            else:
                logits = model(input_ids=iids, attention_mask=amask).logits
                loss   = smooth_ce_loss(logits, lbls) / grad_accum_steps
                loss.backward()
                if step % grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            total_loss += loss.item() * grad_accum_steps

        avg_loss = total_loss / len(tr_loader)

        # ── Validate ──────────────────────────────────────────────────────
        model.eval()
        all_preds, all_true, all_probs = [], [], []

        with torch.no_grad():
            for batch in vl_loader:
                iids  = batch["input_ids"].to(device)
                amask = batch["attention_mask"].to(device)
                out   = model(input_ids=iids, attention_mask=amask)
                probs = torch.softmax(out.logits, dim=1).cpu().numpy()
                all_preds.extend(probs.argmax(axis=1).tolist())
                all_true.extend(batch["label"].numpy().tolist())
                all_probs.extend(probs.tolist())

        from sklearn.metrics import accuracy_score, f1_score
        val_acc = accuracy_score(all_true, all_preds)
        pred_labels = le.inverse_transform(all_preds)
        true_labels = le.inverse_transform(all_true)
        val_wf1 = f1_score(true_labels, pred_labels,
                            average="weighted", zero_division=0)

        history.append({
            "epoch":      epoch,
            "train_loss": round(avg_loss, 4),
            "val_acc":    round(val_acc, 4),
            "val_wf1":    round(val_wf1, 4),
        })

        improved = "✓ best" if val_wf1 > best_val_wf1 else ""
        print(
            f"  Epoch {epoch:2d}/{epochs}  "
            f"loss={avg_loss:.4f}  "
            f"val_acc={val_acc * 100:.1f}%  "
            f"val_wf1={val_wf1:.4f}  {improved}"
        )

        if val_wf1 > best_val_wf1:
            best_val_wf1 = val_wf1
            no_improve   = 0
            os.makedirs(ckpt_dir, exist_ok=True)
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch} "
                      f"(no improvement for {patience} epochs)")
                break

    # ── Final evaluation from best checkpoint ────────────────────────────────
    print(f"\n  Loading best checkpoint (val wf1={best_val_wf1:.4f}) ...")
    from transformers import DistilBertForSequenceClassification as _DBSC
    best_model = _DBSC.from_pretrained(ckpt_dir).to(device)
    best_model.eval()

    final_preds, final_true, final_probs = [], [], []
    with torch.no_grad():
        for batch in vl_loader:
            iids  = batch["input_ids"].to(device)
            amask = batch["attention_mask"].to(device)
            out   = best_model(input_ids=iids, attention_mask=amask)
            probs = torch.softmax(out.logits, dim=1).cpu().numpy()
            final_preds.extend(probs.argmax(axis=1).tolist())
            final_true.extend(batch["label"].numpy().tolist())
            final_probs.extend(probs.tolist())

    pred_labels  = le.inverse_transform(final_preds)
    true_labels  = le.inverse_transform(final_true)
    metrics = compute_metrics(
        true_labels.tolist(), pred_labels.tolist(),
        np.array(final_probs), classes, "DistilBERT",
    )
    metrics["training_history"] = history
    metrics["model_path"]       = ckpt_dir
    metrics["hyperparams"] = {
        "lr": lr, "epochs_run": len(history), "batch_size": batch_size,
        "grad_accum_steps": grad_accum_steps, "max_len": max_len,
        "label_smoothing": label_smoothing, "warmup_ratio": 0.20,
    }

    # Save label encoder and metadata alongside model weights
    with open(os.path.join(ckpt_dir, "label_encoder.pkl"), "wb") as fh:
        pickle.dump(le, fh)
    with open(os.path.join(ckpt_dir, "label_to_group.json"), "w",
              encoding="utf-8") as fh:
        json.dump(LABEL_TO_GROUP, fh, indent=2)

    print(f"  Saved → {ckpt_dir}")
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# RESULT PERSISTENCE  — merge-on-write so individual model runs do not
#                       overwrite results from other already-trained models
# ─────────────────────────────────────────────────────────────────────────────

def _load_results(path: str) -> dict:
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    return {
        "full_results":      {},
        "per_class_results": {},
        "confusion_matrices":{},
    }


def _save_results(data: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


def _merge_into_results(existing: dict, metrics_list: list[dict]) -> dict:
    """Add/overwrite entries for each trained model; preserve untouched models."""
    for m in metrics_list:
        name = m["model_name"]
        existing["full_results"][name] = {
            k: v for k, v in m.items()
            if k not in ("confusion_matrix", "per_class")
        }
        existing["per_class_results"][name]  = m.get("per_class", {})
        existing["confusion_matrices"][name] = {
            "matrix":  m.get("confusion_matrix", []),
            "classes": m.get("classes", []),
        }
    return existing


# ─────────────────────────────────────────────────────────────────────────────
# BEST MODEL SELECTION
# ─────────────────────────────────────────────────────────────────────────────

def select_best(results_this_run: list[dict], save_dir: str = "models/nlp") -> dict:
    """
    Select the best model across ALL trained models (current run + all
    previously trained models stored in intent_results.json).

    Selection metric: weighted_f1 on the val set.
    Reason: the dataset has a 24:1 class imbalance. Weighted F1 weights
    each class by its actual support, reflecting real-world distribution
    far better than macro F1 (which treats a 9-sample class identically
    to a 220-sample class).
    """
    valid_this_run = [r for r in results_this_run if not r.get("skipped")]
    if not valid_this_run:
        print("  No valid results to save.")
        return {}

    results_path = os.path.join(save_dir, "intent_results.json")
    existing     = _load_results(results_path)
    existing     = _merge_into_results(existing, valid_this_run)

    # Collect ALL known models that have a saved file on disk
    all_candidates: list[dict] = []
    for name, data in existing["full_results"].items():
        mpath = data.get("model_path", "")
        # Normalise Windows backslashes
        mpath_norm = mpath.replace("\\", "/")
        if os.path.exists(mpath) or os.path.exists(mpath_norm):
            all_candidates.append(data)

    if not all_candidates:
        all_candidates = valid_this_run

    best = max(all_candidates, key=lambda r: r.get("weighted_f1", 0))

    # ── Print comparison table ────────────────────────────────────────────────
    print(f"\n{'=' * 57}")
    print("  MODEL COMPARISON  (Val Set — all trained models)")
    print(f"{'=' * 57}")
    print(f"  {'Model':<28} {'Acc':>7} {'MacF1':>8} {'WtdF1':>8} {'Unc%':>7}")
    print("  " + "─" * 57)
    for r in sorted(all_candidates, key=lambda x: -x.get("weighted_f1", 0)):
        unc = f"{r['uncertainty_rate'] * 100:.1f}" \
            if r.get("uncertainty_rate") is not None else "N/A"
        marker = " ←" if r["model_name"] == best["model_name"] else ""
        print(
            f"  {r['model_name']:<28}"
            f"{r['accuracy'] * 100:>6.1f}%"
            f"{r['macro_f1']:>9.4f}"
            f"{r['weighted_f1']:>9.4f}"
            f"{unc:>8}{marker}"
        )
    print(f"\n  ✓ Best: {best['model_name']}  "
          f"(weighted_f1={best['weighted_f1']:.4f})")

    summary = {
        "best_model":        best["model_name"],
        "best_model_path":   best.get("model_path", ""),
        "selection_metric":  "weighted_f1",
        "selection_reason": (
            "Weighted F1 is used because the dataset has a 24:1 class imbalance. "
            "It weights each class by its support in the val set, reflecting the "
            "real-world distribution better than macro F1."
        ),
        "comparison": [
            {
                "model":            r["model_name"],
                "accuracy":         r.get("accuracy"),
                "macro_f1":         r.get("macro_f1"),
                "weighted_f1":      r.get("weighted_f1"),
                "macro_precision":  r.get("macro_precision"),
                "macro_recall":     r.get("macro_recall"),
                "uncertainty_rate": r.get("uncertainty_rate"),
                "avg_confidence":   r.get("avg_confidence"),
            }
            for r in all_candidates
        ],
        "intent_to_action":      INTENT_TO_ACTION,
        "intent_groups":         INTENT_GROUPS,
        "uncertainty_threshold": UNCERTAINTY_THRESHOLD,
    }

    existing["summary"] = summary
    _save_results(existing, results_path)

    best_path = os.path.join(save_dir, "best_model.json")
    with open(best_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    print(f"  Saved → {results_path}")
    print(f"  Saved → {best_path}")
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# TEST SET EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_on_test(
    model_name: str,
    test_df: pd.DataFrame,
    save_dir: str = "models/nlp",
) -> dict:
    """
    Run the specified trained model on the held-out test set.
    Results are merged into intent_results.json under 'test_results'.

    Call this ONLY after all training and val-based tuning is complete.
    """
    print(f"\n{'=' * 57}")
    print(f"  TEST SET  |  {model_name}")
    print(f"{'=' * 57}")

    y_true = map_labels(test_df["primary_label"])

    if model_name == "logreg":
        with open(os.path.join(save_dir, "intent_tfidf_logreg.pkl"), "rb") as fh:
            obj = pickle.load(fh)
        y_pred  = obj["pipeline"].predict(test_df["clean_text"].fillna(""))
        y_prob  = obj["pipeline"].predict_proba(test_df["clean_text"].fillna(""))
        classes = obj["classes"]

    elif model_name == "svm":
        with open(os.path.join(save_dir, "intent_tfidf_svm.pkl"), "rb") as fh:
            obj = pickle.load(fh)
        y_pred  = obj["pipeline"].predict(test_df["clean_text"].fillna(""))
        y_prob  = obj["pipeline"].predict_proba(test_df["clean_text"].fillna(""))
        classes = obj["classes"]

    elif model_name == "distilbert":
        try:
            import torch
            from transformers import (
                DistilBertForSequenceClassification,
                DistilBertTokenizerFast,
            )
        except ImportError as exc:
            raise ImportError("pip install transformers torch") from exc

        ckpt   = os.path.join(save_dir, "intent_distilbert")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tok    = DistilBertTokenizerFast.from_pretrained(ckpt)
        model  = DistilBertForSequenceClassification.from_pretrained(ckpt).to(device)
        model.eval()
        with open(os.path.join(ckpt, "label_encoder.pkl"), "rb") as fh:
            le = pickle.load(fh)
        classes = le.classes_.tolist()

        preds, probs_all = [], []
        for text in test_df["clean_text"].fillna("").astype(str):
            enc = tok(
                text, max_length=128, truncation=True,
                padding="max_length", return_tensors="pt",
            )
            with torch.no_grad():
                out  = model(
                    input_ids=enc["input_ids"].to(device),
                    attention_mask=enc["attention_mask"].to(device),
                )
                prob = torch.softmax(out.logits, dim=1).cpu().numpy()[0]
            preds.append(le.inverse_transform([prob.argmax()])[0])
            probs_all.append(prob.tolist())
        y_pred = preds
        y_prob = np.array(probs_all)

    else:
        raise ValueError(
            f"Unknown model '{model_name}'. Choose: logreg | svm | distilbert"
        )

    metrics = compute_metrics(
        y_true.tolist(),
        y_pred if isinstance(y_pred, list) else y_pred.tolist(),
        y_prob,
        classes,
        f"{model_name} [TEST]",
    )

    results_path = os.path.join(save_dir, "intent_results.json")
    existing     = _load_results(results_path)
    existing.setdefault("test_results", {})[model_name] = {
        k: v for k, v in metrics.items()
        if k not in ("confusion_matrix", "per_class")
    }
    existing.setdefault("test_confusion_matrices", {})[model_name] = {
        "matrix":  metrics.get("confusion_matrix", []),
        "classes": metrics.get("classes", []),
    }
    existing.setdefault("test_per_class", {})[model_name] = \
        metrics.get("per_class", {})
    _save_results(existing, results_path)
    print(f"  Test results saved → {results_path}")
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE  — used by nlp_pipeline.py and confidence_estimator.py
# ─────────────────────────────────────────────────────────────────────────────

def predict(text: str, save_dir: str = "models/nlp") -> dict:
    """
    Classify a single issue text using the best saved model.

    Returns:
        intent_group     — predicted intent (9 classes)
        confidence_score — max predicted class probability [0, 1]
        uncertainty_flag — 1 if confidence < UNCERTAINTY_THRESHOLD
        suggested_action — RL action recommendation
        top3_predictions — top-3 classes with probabilities
        model_used       — which model made the prediction
    """
    best_path = os.path.join(save_dir, "best_model.json")
    if not os.path.exists(best_path):
        raise FileNotFoundError(
            f"No trained model found at {best_path}.\n"
            "Run: python nlp/intent_classifier.py --model all"
        )

    with open(best_path, encoding="utf-8") as fh:
        best = json.load(fh)

    mname = best["best_model"]
    mpath = os.path.normpath(best["best_model_path"])
    # Normalise Windows backslashes for cross-platform compatibility
    mpath = mpath.replace("\\", "/")

    if "DistilBERT" in mname:
        import torch
        from transformers import (
            DistilBertForSequenceClassification,
            DistilBertTokenizerFast,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tok    = DistilBertTokenizerFast.from_pretrained(mpath)
        model  = DistilBertForSequenceClassification.from_pretrained(mpath).to(device)
        model.eval()
        with open(os.path.join(mpath, "label_encoder.pkl"), "rb") as fh:
            le = pickle.load(fh)
        enc = tok(
            text, max_length=128, truncation=True,
            padding="max_length", return_tensors="pt",
        )
        with torch.no_grad():
            out  = model(
                input_ids=enc["input_ids"].to(device),
                attention_mask=enc["attention_mask"].to(device),
            )
            prob = torch.softmax(out.logits, dim=1).cpu().numpy()[0]
        pred       = le.inverse_transform([prob.argmax()])[0]
        confidence = float(prob.max())
        top3 = [
            {"label": le.inverse_transform([i])[0], "prob": round(float(prob[i]), 4)}
            for i in prob.argsort()[-3:][::-1]
        ]

    else:  # TF-IDF models
        pkl_name = (
            "intent_tfidf_svm.pkl"
            if "SVM" in mname
            else "intent_tfidf_logreg.pkl"
        )
        # Use save_dir parameter, but also try the directory of best_model_path
        # as a fallback. This handles cases where the notebook runs from a
        # different working directory than where models were saved.
        candidates = [
            os.path.join(save_dir, pkl_name),
            os.path.join(os.path.dirname(mpath), pkl_name),
            os.path.join(os.path.dirname(mpath.replace("\\", "/")), pkl_name),
        ]
        pkl_path = next((p for p in candidates if os.path.exists(p)), None)
        if pkl_path is None:
            raise FileNotFoundError(
                f"Cannot find {pkl_name}.\n"
                f"Tried:\n" + "\n".join(f"  {p}" for p in candidates) +
                "\nMake sure you are running from the project root."
            )
        with open(pkl_path, "rb") as fh:
            obj = pickle.load(fh)
        pipeline   = obj["pipeline"]
        prob       = pipeline.predict_proba([text])[0]
        classes    = pipeline.classes_
        idx        = prob.argmax()
        pred       = classes[idx]
        confidence = float(prob[idx])
        top3 = [
            {"label": classes[i], "prob": round(float(prob[i]), 4)}
            for i in prob.argsort()[-3:][::-1]
        ]

    return {
        "intent_group":     pred,
        "confidence_score": round(confidence, 4),
        "uncertainty_flag": int(confidence < UNCERTAINTY_THRESHOLD),
        "suggested_action": INTENT_TO_ACTION.get(pred, "clarify"),
        "top3_predictions": top3,
        "model_used":       mname,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="NexResolve Intent Classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", default="all",
        choices=["all", "logreg", "svm", "distilbert"],
        help="Which model(s) to train / evaluate / predict with",
    )
    parser.add_argument(
        "--mode", default="train",
        choices=["train", "test", "predict"],
    )
    parser.add_argument("--text",             default=None,
                        help="Issue text for --mode predict")
    parser.add_argument("--epochs",           type=int,   default=8)
    parser.add_argument("--batch_size",       type=int,   default=16)
    parser.add_argument("--max_len",          type=int,   default=128)
    parser.add_argument("--lr",               type=float, default=3e-5)
    parser.add_argument("--grad_accum_steps", type=int,   default=1)
    parser.add_argument("--label_smoothing",  type=float, default=0.0)
    parser.add_argument("--save_dir",         default="models/nlp")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # ── Predict ───────────────────────────────────────────────────────────────
    if args.mode == "predict":
        text   = args.text or input("Issue text: ").strip()
        result = predict(text, args.save_dir)
        print(json.dumps(result, indent=2))
        return

    # ── Test ──────────────────────────────────────────────────────────────────
    if args.mode == "test":
        test_df = pd.read_csv("data/splits/test.csv")
        if args.model == "all":
            for m in ["logreg", "svm", "distilbert"]:
                try:
                    evaluate_on_test(m, test_df, args.save_dir)
                except FileNotFoundError as exc:
                    print(f"  SKIP {m}: {exc}")
        else:
            evaluate_on_test(args.model, test_df, args.save_dir)
        return

    # ── Train ─────────────────────────────────────────────────────────────────
    print("Loading data ...")
    train_df = pd.read_csv("data/splits/train.csv")
    val_df   = pd.read_csv("data/splits/val.csv")
    print(f"  Train: {len(train_df)} | Val: {len(val_df)}")

    trained: list[dict] = []

    if args.model in ("all", "logreg"):
        trained.append(train_logreg(train_df, val_df, args.save_dir))

    if args.model in ("all", "svm"):
        trained.append(train_svm(train_df, val_df, args.save_dir))

    if args.model in ("all", "distilbert"):
        trained.append(train_distilbert(
            train_df, val_df, args.save_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            max_len=args.max_len,
            lr=args.lr,
            grad_accum_steps=args.grad_accum_steps,
            label_smoothing=args.label_smoothing,
        ))

    if trained:
        select_best(trained, args.save_dir)

    print("\n✓ Training complete.")
    print("  Next steps:")
    print("    python nlp/intent_classifier.py --mode test --model all")
    print("    python evaluation/intent_eval.py --verbose")
    print("    python nlp/confidence_estimator.py --mode generate --model best")
    print("    jupyter notebook notebooks/03_intent_classification.ipynb")


if __name__ == "__main__":
    main()