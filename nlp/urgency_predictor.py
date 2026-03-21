

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

URGENT_THRESHOLD: float = 0.50   # default; updated by threshold analysis

# Structured features used by ML models
FEATURE_COLS: list[str] = [
    "urgency_keyword_flag",
    "word_count",
    "question_mark_flag",
    "missing_error_flag",
    "missing_version_flag",
    "sla_limit_hours",
    "text_length",
    "turn_count",
    "reassignment_count",
    "reopen_count",
    "frustration_score",
]

# Urgency keyword tiers with weights
# Higher weight = stronger urgency signal
URGENCY_KEYWORDS: dict[str, float] = {
    # Critical — service affecting
    "crash":        0.40,
    "outage":       0.40,
    "down":         0.30,
    "production":   0.25,
    "blocker":      0.35,
    "critical":     0.35,
    # High urgency
    "urgent":       0.30,
    "broken":       0.25,
    "stuck":        0.20,
    "freeze":       0.20,
    "cannot":       0.15,
    "rate limit":   0.20,
    "rate-limit":   0.20,
    # Medium urgency
    "regression":   0.15,
    "failed":       0.10,
    "error":        0.10,
    "exception":    0.10,
}


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv("data/splits/train.csv")
    val   = pd.read_csv("data/splits/val.csv")
    test  = pd.read_csv("data/splits/test.csv")
    return train, val, test


def _get_X(df: pd.DataFrame) -> pd.DataFrame:
    """Extract and fill feature matrix."""
    return df[FEATURE_COLS].fillna(0).astype(float)


def _get_y(df: pd.DataFrame) -> pd.Series:
    return df["sla_breach_flag"].astype(int)


def _load_results(path: str) -> dict:
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    return {"full_results": {}, "per_model_results": {}}


def _save_results(data: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION — all binary classification metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
) -> dict:
    """
    Compute all relevant binary classification metrics.

    Metrics:
      accuracy          — overall fraction correct
      precision         — of predicted urgent, how many were truly urgent
      recall            — of truly urgent, how many did we catch
      f1                — harmonic mean of precision and recall (PRIMARY METRIC)
      roc_auc           — area under ROC curve
      avg_precision     — area under precision-recall curve
      confusion_matrix  — [[TN, FP], [FN, TP]]
      threshold_used    — what threshold converted score → flag
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, average_precision_score,
        confusion_matrix,
    )

    acc  = accuracy_score(y_true, y_pred)
    pre  = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    auc  = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    ap   = average_precision_score(y_true, y_prob)
    cm   = confusion_matrix(y_true, y_pred).tolist()

    # Per-class breakdown
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    result = {
        "model_name":       model_name,
        "accuracy":         round(acc, 4),
        "precision":        round(pre, 4),
        "recall":           round(rec, 4),
        "f1":               round(f1,  4),
        "roc_auc":          round(auc, 4),
        "avg_precision":    round(ap,  4),
        "confusion_matrix": cm,
        "true_positives":   int(tp),
        "false_positives":  int(fp),
        "true_negatives":   int(tn),
        "false_negatives":  int(fn),
        "urgent_predicted_pct": round(float(y_pred.mean()) * 100, 2),
    }

    _print_metrics(result)
    return result


def _print_metrics(r: dict) -> None:
    sep = "=" * 57
    print(f"\n{sep}")
    print(f"  {r['model_name']}")
    print(sep)
    print(f"  Accuracy:          {r['accuracy'] * 100:>6.2f}%")
    print(f"  Precision:         {r['precision']:>8.4f}  (of flagged urgent, % truly urgent)")
    print(f"  Recall:            {r['recall']:>8.4f}  (of truly urgent, % caught)  ← key")
    print(f"  F1 Score:          {r['f1']:>8.4f}  ← selection metric")
    print(f"  ROC-AUC:           {r['roc_auc']:>8.4f}")
    print(f"  Avg Precision:     {r['avg_precision']:>8.4f}")
    print(f"  Urgent predicted:  {r['urgent_predicted_pct']:>6.1f}%")
    print()
    cm = r["confusion_matrix"]
    print("  Confusion Matrix:")
    print(f"                  Predicted")
    print(f"             Not-Urgent  Urgent")
    print(f"  True Not-Urgent  {cm[0][0]:>5}   {cm[0][1]:>5}")
    print(f"  True Urgent      {cm[1][0]:>5}   {cm[1][1]:>5}")
    print(f"  (FP={r['false_positives']}  FN={r['false_negatives']}  "
          f"TP={r['true_positives']}  TN={r['true_negatives']})")


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 1 — Rule-based keyword scoring
# ─────────────────────────────────────────────────────────────────────────────

def train_rule_based(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    save_dir: str = "models/nlp",
    threshold: float = URGENT_THRESHOLD,
) -> dict:
    """
    No training — pure logic baseline.

    Scoring logic:
      base score = 0.30  (any ticket has baseline urgency)
      + keyword weights  (crash=+0.40, urgent=+0.30, error=+0.10, ...)
      + sla_limit <= 24h → +0.20  (tight SLA = urgent)
      + reassignment > 0 → +0.10  (reassigned = complicated)
      + reopen > 0       → +0.15  (reopened = unresolved previously)
      clipped to [0, 1]

    This model is interpretable and requires no data.
    It serves as the lower bound for ML models to beat.
    """
    print("\n" + "─" * 57)
    print("  MODEL 1  |  Rule-Based Keyword Scoring")
    print("─" * 57)

    def score_df(df: pd.DataFrame) -> np.ndarray:
        texts  = df["clean_text"].fillna("").str.lower()
        scores = np.full(len(df), 0.30)

        # Keyword signals
        for kw, weight in URGENCY_KEYWORDS.items():
            mask    = texts.str.contains(kw, regex=False)
            scores += mask.values * weight

        # SLA tier signal
        if "sla_limit_hours" in df.columns:
            tight_sla = (df["sla_limit_hours"].fillna(72) <= 24).values
            scores   += tight_sla * 0.20

        # Interaction signals
        if "reassignment_count" in df.columns:
            scores += (df["reassignment_count"].fillna(0) > 0).values * 0.10
        if "reopen_count" in df.columns:
            scores += (df["reopen_count"].fillna(0) > 0).values * 0.15

        return np.clip(scores, 0.0, 1.0)

    # Evaluate on val
    y_vl    = _get_y(val_df).values
    vl_prob = score_df(val_df)
    vl_pred = (vl_prob >= threshold).astype(int)
    metrics = compute_metrics(y_vl, vl_pred, vl_prob, "Rule-Based")
    metrics["threshold_used"] = threshold

    # Save scorer function via pickle-friendly object
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "urgency_rule_based.pkl")
    with open(path, "wb") as fh:
        pickle.dump({
            "type":      "rule_based",
            "keywords":  URGENCY_KEYWORDS,
            "threshold": threshold,
        }, fh)
    print(f"  Saved → {path}")
    metrics["model_path"] = path
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 2 — Logistic Regression on structured features
# ─────────────────────────────────────────────────────────────────────────────

def train_logreg(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    save_dir: str = "models/nlp",
    threshold: float = URGENT_THRESHOLD,
) -> dict:
    """
    Logistic Regression on structured features only.
    Interpretable — coefficients show which features matter most.
    StandardScaler normalises features before fitting.
    class_weight='balanced' handles the mild imbalance.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    print("\n" + "─" * 57)
    print("  MODEL 2  |  Logistic Regression (structured features)")
    print("─" * 57)

    X_tr = _get_X(train_df)
    y_tr = _get_y(train_df)
    X_vl = _get_X(val_df)
    y_vl = _get_y(val_df)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
            solver="lbfgs",
        )),
    ])

    print("  Training ...")
    pipeline.fit(X_tr, y_tr)

    vl_prob = pipeline.predict_proba(X_vl)[:, 1]
    vl_pred = (vl_prob >= threshold).astype(int)
    metrics = compute_metrics(y_vl.values, vl_pred, vl_prob, "Logistic Regression")
    metrics["threshold_used"] = threshold

    # Feature coefficients (interpretability)
    coefs = dict(zip(
        FEATURE_COLS,
        pipeline.named_steps["clf"].coef_[0].round(4).tolist()
    ))
    metrics["feature_coefficients"] = coefs
    print("  Top feature coefficients:")
    for feat, coef in sorted(coefs.items(), key=lambda x: -abs(x[1]))[:6]:
        direction = "↑urgent" if coef > 0 else "↓urgent"
        print(f"    {feat:30s}: {coef:>+7.4f}  {direction}")

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "urgency_logreg.pkl")
    with open(path, "wb") as fh:
        pickle.dump({
            "pipeline":    pipeline,
            "feature_cols": FEATURE_COLS,
            "threshold":   threshold,
        }, fh)
    print(f"  Saved → {path}")
    metrics["model_path"] = path
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 3 — Gradient Boosting
# ─────────────────────────────────────────────────────────────────────────────

def train_gradboost(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    save_dir: str = "models/nlp",
    threshold: float = URGENT_THRESHOLD,
) -> dict:
    """
    Gradient Boosting Classifier on structured features.
    Best performing model — captures non-linear feature interactions.

    Why GradBoost over XGBoost?
      sklearn's GradientBoostingClassifier requires no extra install.
      Performance is equivalent on this dataset size (596 train).

    Key interactions it captures:
      sla_limit_hours × urgency_keyword  — tight SLA + urgent keyword = very urgent
      turn_count × frustration_score     — long back-and-forth + frustrated = urgent
      reassignment + reopen              — repeatedly unresolved = very urgent
    """
    from sklearn.ensemble import GradientBoostingClassifier

    print("\n" + "─" * 57)
    print("  MODEL 3  |  Gradient Boosting (structured features)")
    print("─" * 57)

    X_tr = _get_X(train_df)
    y_tr = _get_y(train_df)
    X_vl = _get_X(val_df)
    y_vl = _get_y(val_df)

    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=5,
        random_state=42,
    )

    print("  Training (200 estimators) ...")
    model.fit(X_tr, y_tr)

    vl_prob = model.predict_proba(X_vl)[:, 1]
    vl_pred = (vl_prob >= threshold).astype(int)
    metrics = compute_metrics(y_vl.values, vl_pred, vl_prob, "Gradient Boosting")
    metrics["threshold_used"] = threshold

    # Feature importances
    importances = dict(zip(
        FEATURE_COLS,
        model.feature_importances_.round(4).tolist()
    ))
    metrics["feature_importances"] = importances
    print("  Feature importances:")
    for feat, imp in sorted(importances.items(), key=lambda x: -x[1]):
        bar = "█" * int(imp * 40)
        print(f"    {feat:30s}: {imp:.4f}  |{bar}")

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "urgency_gradboost.pkl")
    with open(path, "wb") as fh:
        pickle.dump({
            "model":       model,
            "feature_cols": FEATURE_COLS,
            "threshold":   threshold,
        }, fh)
    print(f"  Saved → {path}")
    metrics["model_path"] = path
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# THRESHOLD ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def analyse_threshold(
    model_name: str = "gradboost",
    save_dir: str   = "models/nlp",
    val_df: Optional[pd.DataFrame] = None,
    thresholds: Optional[list] = None,
) -> dict:
    """
    Find optimal threshold for converting urgency_score → urgent_flag.

    For urgency, urgent_flag = 1 when urgency_score >= threshold.
    Selection criterion: F1 on breach class (class=1).

    Unlike intent (where we maximised signal),
    here we maximise F1 because:
      - Missing an urgent ticket (FN) is worse than over-flagging (FP)
      - F1 balances catching urgents (recall) vs. precision
      - The threshold that maximises F1 is the operating point for RL

    Saved to: models/nlp/urgency_threshold_analysis.json
    """
    from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

    if thresholds is None:
        thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]

    if val_df is None:
        val_df = pd.read_csv("data/splits/val.csv")

    y_true = _get_y(val_df).values

    # Get probabilities from saved model
    probs = _get_probs(model_name, val_df, save_dir)

    results = {}
    print(f"\n[threshold_analysis]  model={model_name}")
    print(f"  {'T':>5} | {'Prec':>7} {'Rec':>7} {'F1':>7} {'Acc':>7} | {'Urgent%':>8}")
    print("  " + "─" * 55)

    # True breach rate on val set — used to penalise over-flagging
    true_breach_rate = float(y_true.mean())

    best_f1 = 0.0
    best_t  = 0.5

    for t in thresholds:
        preds = (probs >= t).astype(int)
        f1    = f1_score(y_true, preds, zero_division=0)
        pre   = precision_score(y_true, preds, zero_division=0)
        rec   = recall_score(y_true, preds, zero_division=0)
        acc   = accuracy_score(y_true, preds)
        pct   = preds.mean() * 100
        over_flag_pct = (preds.mean() - true_breach_rate) * 100

        results[str(t)] = {
            "threshold":      t,
            "f1":             round(f1,  4),
            "precision":      round(pre, 4),
            "recall":         round(rec, 4),
            "accuracy":       round(acc, 4),
            "urgent_pct":     round(pct, 2),
            "over_flag_pct":  round(over_flag_pct, 2),
        }

        marker = " ← best" if f1 > best_f1 else ""
        print(
            f"  {t:>5.2f} | {pre:>7.3f} {rec:>7.3f} {f1:>7.3f} "
            f"{acc:>7.3f} | {pct:>7.1f}% ({over_flag_pct:+.1f}%){marker}"
        )

        # Selection criterion:
        # 1. Maximise F1 on breach class
        # 2. But discard thresholds that over-flag by more than 8%
        #    (i.e. predicted urgent% exceeds true breach rate by > 8 points)
        #    because too much over-flagging floods the RL agent with false urgency
        within_overflg_limit = over_flag_pct <= 8.0
        if f1 > best_f1 and within_overflg_limit:
            best_f1 = f1
            best_t  = t

    # Fallback: if all thresholds exceed 8% over-flag, just pick best F1
    if best_t == 0.5 and best_f1 == 0.0:
        best_t = max(results, key=lambda k: results[k]["f1"])
        best_f1 = results[best_t]["f1"]
        best_t  = float(best_t)

    print(f"\n  True breach rate (val): {true_breach_rate*100:.1f}%")
    print(f"  Recommended threshold:  {best_t}  "
          f"(F1={best_f1:.4f}, over-flag <= 8%)")

    out = {
        "model_type":             model_name,
        "per_threshold":          results,
        "recommended_threshold":  best_t,
        "recommended_f1":         round(best_f1, 4),
        "selection_criterion":    "F1 on breach class (class=1)",
    }
    path = os.path.join(save_dir, "urgency_threshold_analysis.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"  Saved → {path}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# BEST MODEL SELECTION
# ─────────────────────────────────────────────────────────────────────────────

def select_best(results_this_run: list[dict], save_dir: str = "models/nlp") -> dict:
    """
    Select best model across all trained models using F1 on val set.
    Merges with previously saved results so re-running one model
    does not erase others.
    """
    valid = [r for r in results_this_run if not r.get("skipped")]
    if not valid:
        print("  No valid results.")
        return {}

    results_path = os.path.join(save_dir, "urgency_results.json")
    existing     = _load_results(results_path)

    # Merge new results into existing
    for r in valid:
        name = r["model_name"]
        existing["full_results"][name] = {
            k: v for k, v in r.items()
            if k != "confusion_matrix"
        }
        existing["per_model_results"][name] = r

    # Compare all known models
    all_candidates = []
    for name, data in existing["full_results"].items():
        mpath = data.get("model_path", "")
        if os.path.exists(mpath) or os.path.exists(mpath.replace("\\", "/")):
            all_candidates.append(data)

    if not all_candidates:
        all_candidates = valid

    best = max(all_candidates, key=lambda r: r.get("f1", 0))

    # Print comparison table
    print(f"\n{'=' * 57}")
    print("  URGENCY MODEL COMPARISON  (Val Set)")
    print(f"{'=' * 57}")
    print(f"  {'Model':<25} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AUC':>7}")
    print("  " + "─" * 57)
    for r in sorted(all_candidates, key=lambda x: -x.get("f1", 0)):
        marker = " ←" if r["model_name"] == best["model_name"] else ""
        print(
            f"  {r['model_name']:<25}"
            f"{r['accuracy']*100:>6.1f}%"
            f"{r.get('precision',0):>8.4f}"
            f"{r.get('recall',0):>8.4f}"
            f"{r.get('f1',0):>8.4f}"
            f"{r.get('roc_auc',0):>8.4f}{marker}"
        )
    print(f"\n  ✓ Best: {best['model_name']}  (F1={best['f1']:.4f})")

    summary = {
        "best_model":       best["model_name"],
        "best_model_path":  best.get("model_path", ""),
        "selection_metric": "f1",
        "selection_reason": (
            "F1 on breach class chosen because missing an urgent ticket "
            "(false negative) is worse than over-flagging a normal one. "
            "F1 balances precision and recall for the urgent class."
        ),
        "comparison": [
            {
                "model":        r["model_name"],
                "accuracy":     r.get("accuracy"),
                "precision":    r.get("precision"),
                "recall":       r.get("recall"),
                "f1":           r.get("f1"),
                "roc_auc":      r.get("roc_auc"),
                "avg_precision":r.get("avg_precision"),
            }
            for r in all_candidates
        ],
        "urgent_threshold":     URGENT_THRESHOLD,
        "feature_cols":         FEATURE_COLS,
    }

    existing["summary"] = summary
    _save_results(existing, results_path)

    best_path = os.path.join(save_dir, "urgency_best_model.json")
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
    Evaluate on held-out test set.
    Call only after all training and threshold tuning is done.
    """
    print(f"\n{'=' * 57}")
    print(f"  TEST SET  |  {model_name}")
    print(f"{'=' * 57}")

    # Load threshold from threshold analysis if available
    ta_path   = os.path.join(save_dir, "urgency_threshold_analysis.json")
    threshold = URGENT_THRESHOLD
    if os.path.exists(ta_path):
        with open(ta_path, encoding="utf-8") as fh:
            ta = json.load(fh)
        threshold = ta.get("recommended_threshold", URGENT_THRESHOLD)

    y_true = _get_y(test_df).values
    probs  = _get_probs(model_name, test_df, save_dir)
    preds  = (probs >= threshold).astype(int)

    metrics = compute_metrics(y_true, preds, probs, f"{model_name} [TEST]")
    metrics["threshold_used"] = threshold

    # Append to results file
    results_path = os.path.join(save_dir, "urgency_results.json")
    existing = _load_results(results_path)
    existing.setdefault("test_results", {})[model_name] = {
        k: v for k, v in metrics.items() if k != "confusion_matrix"
    }
    existing.setdefault("test_confusion_matrices", {})[model_name] = \
        metrics.get("confusion_matrix", [])
    _save_results(existing, results_path)
    print(f"  Test results saved → {results_path}")
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# GENERATE FEATURES — write urgency_score + urgent_flag to CSVs
# ─────────────────────────────────────────────────────────────────────────────

def generate_features(
    model_name: str  = "best",
    save_dir: str    = "models/nlp",
    issues_path: str = "data/final/cleaned_issues.csv",
    rl_path: str     = "data/final/final_rl_dataset.csv",
) -> pd.DataFrame:
    """
    Run the best urgency model on all 852 issues.
    Writes urgency_score and urgent_flag to:
      data/final/cleaned_issues.csv
      data/final/final_rl_dataset.csv

    These columns are used by the RL state builder.
    """
    if model_name == "best":
        best_path = os.path.join(save_dir, "urgency_best_model.json")
        if not os.path.exists(best_path):
            raise FileNotFoundError(
                "No urgency_best_model.json. Train models first."
            )
        with open(best_path, encoding="utf-8") as fh:
            best = json.load(fh)
        model_name = best["best_model"]
        print(f"[generate_features] Best model: {model_name}")

    # Load threshold
    ta_path   = os.path.join(save_dir, "urgency_threshold_analysis.json")
    threshold = URGENT_THRESHOLD
    if os.path.exists(ta_path):
        with open(ta_path, encoding="utf-8") as fh:
            ta = json.load(fh)
        threshold = ta.get("recommended_threshold", URGENT_THRESHOLD)
    print(f"[generate_features] Threshold: {threshold}")

    ci = pd.read_csv(issues_path, encoding="utf-8")
    print(f"  Processing {len(ci)} issues ...")

    # Some FEATURE_COLS (sla_limit_hours, turn_count, reassignment_count,
    # reopen_count, frustration_score) only exist in final_rl_dataset.csv.
    # Merge them in before running the model.
    rl_cols_needed = [c for c in FEATURE_COLS if c not in ci.columns]
    if rl_cols_needed:
        rl_tmp = pd.read_csv(rl_path, encoding="utf-8")
        merge_cols = ["issue_number"] + [c for c in rl_cols_needed if c in rl_tmp.columns]
        ci = ci.merge(rl_tmp[merge_cols], on="issue_number", how="left")
        print(f"  Merged {len(rl_cols_needed)} RL columns from final_rl_dataset.csv")

    probs = _get_probs(model_name, ci, save_dir)

    ci["urgency_score"] = probs.round(4)
    ci["urgent_flag"]   = (probs >= threshold).astype(int)

    # Drop old columns before writing RL dataset
    rl = pd.read_csv(rl_path, encoding="utf-8")
    for col in ["urgency_score", "urgent_flag"]:
        if col in rl.columns:
            rl = rl.drop(columns=[col])

    ci.to_csv(issues_path, index=False, encoding="utf-8")
    print(f"  Updated → {issues_path}")

    rl = rl.merge(
        ci[["issue_number", "urgency_score", "urgent_flag"]],
        on="issue_number", how="left"
    )
    rl.to_csv(rl_path, index=False, encoding="utf-8")
    print(f"  Updated → {rl_path}")

    urgent_pct = ci["urgent_flag"].mean() * 100
    print(f"\n  Urgent (flag=1): {ci['urgent_flag'].sum()} ({urgent_pct:.1f}%)")
    print(f"  Avg urgency score: {ci['urgency_score'].mean():.4f}")
    return ci[["issue_number", "urgency_score", "urgent_flag"]]


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

def _get_probs(
    model_name: str,
    df: pd.DataFrame,
    save_dir: str = "models/nlp",
) -> np.ndarray:
    """Load model and return urgency_score probabilities for df."""
    model_name_lower = model_name.lower().replace(" ", "_").replace("-", "_")

    if "rule" in model_name_lower:
        def score_df(d):
            texts  = d["clean_text"].fillna("").str.lower()
            scores = np.full(len(d), 0.30)
            for kw, weight in URGENCY_KEYWORDS.items():
                scores += texts.str.contains(kw, regex=False).values * weight
            if "sla_limit_hours" in d.columns:
                scores += (d["sla_limit_hours"].fillna(72) <= 24).values * 0.20
            if "reassignment_count" in d.columns:
                scores += (d["reassignment_count"].fillna(0) > 0).values * 0.10
            if "reopen_count" in d.columns:
                scores += (d["reopen_count"].fillna(0) > 0).values * 0.15
            return np.clip(scores, 0.0, 1.0)
        return score_df(df)

    elif "logreg" in model_name_lower or "logistic" in model_name_lower:
        path = os.path.normpath(os.path.join(save_dir, "urgency_logreg.pkl"))
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        return obj["pipeline"].predict_proba(_get_X(df))[:, 1]

    elif "grad" in model_name_lower or "boost" in model_name_lower:
        path = os.path.normpath(os.path.join(save_dir, "urgency_gradboost.pkl"))
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        return obj["model"].predict_proba(_get_X(df))[:, 1]

    else:
        raise ValueError(
            f"Unknown model: '{model_name}'. "
            "Choose: rule_based | logreg | gradboost | best"
        )


def predict(text: str, structured: Optional[dict] = None,
            save_dir: str = "models/nlp") -> dict:
    """
    Predict urgency for a single ticket.

    Args:
        text:       clean_text of the issue
        structured: optional dict with numeric features
                    e.g. {'sla_limit_hours': 24, 'turn_count': 3}
        save_dir:   model directory

    Returns:
        urgency_score, urgent_flag, threshold_used, model_used
    """
    best_path = os.path.join(save_dir, "urgency_best_model.json")
    if not os.path.exists(best_path):
        raise FileNotFoundError(
            "No urgency_best_model.json. "
            "Run: python nlp/urgency_predictor.py --model all"
        )
    with open(best_path, encoding="utf-8") as fh:
        best = json.load(fh)

    ta_path   = os.path.join(save_dir, "urgency_threshold_analysis.json")
    threshold = URGENT_THRESHOLD
    if os.path.exists(ta_path):
        with open(ta_path, encoding="utf-8") as fh:
            ta = json.load(fh)
        threshold = ta.get("recommended_threshold", URGENT_THRESHOLD)

    mname = best["best_model"]

    # Build a single-row DataFrame
    row = {col: 0.0 for col in FEATURE_COLS}
    row["clean_text"] = text
    if structured:
        for k, v in structured.items():
            if k in row:
                row[k] = v

    # Keyword flag from text
    text_lower = text.lower()
    row["urgency_keyword_flag"] = int(
        any(kw in text_lower for kw in URGENCY_KEYWORDS)
    )

    df_single = pd.DataFrame([row])
    probs     = _get_probs(mname, df_single, save_dir)
    score     = float(probs[0])

    return {
        "urgency_score":  round(score, 4),
        "urgent_flag":    int(score >= threshold),
        "threshold_used": threshold,
        "model_used":     mname,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="NexResolve Urgency Predictor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default="all",
                        choices=["all","rule_based","logreg","gradboost"])
    parser.add_argument("--mode",  default="train",
                        choices=["train","test","generate","predict","threshold"])
    parser.add_argument("--text",     default=None)
    parser.add_argument("--save_dir", default="models/nlp")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    if args.mode == "predict":
        text   = args.text or input("Issue text: ").strip()
        result = predict(text, save_dir=args.save_dir)
        print(json.dumps(result, indent=2))
        return

    if args.mode == "threshold":
        _, val_df, _ = _load_splits()
        model = args.model if args.model != "all" else "gradboost"
        analyse_threshold(model, args.save_dir, val_df)
        return

    if args.mode == "generate":
        generate_features(save_dir=args.save_dir)
        return

    if args.mode == "test":
        _, _, test_df = _load_splits()
        models = (["rule_based","logreg","gradboost"]
                  if args.model == "all" else [args.model])
        for m in models:
            try:
                evaluate_on_test(m, test_df, args.save_dir)
            except FileNotFoundError as exc:
                print(f"  SKIP {m}: {exc}")
        return

    # ── Train ──────────────────────────────────────────────────────────────
    print("Loading data ...")
    train_df, val_df, _ = _load_splits()
    print(f"  Train: {len(train_df)} | Val: {len(val_df)}")

    trained: list[dict] = []

    if args.model in ("all", "rule_based"):
        trained.append(train_rule_based(train_df, val_df, args.save_dir))
    if args.model in ("all", "logreg"):
        trained.append(train_logreg(train_df, val_df, args.save_dir))
    if args.model in ("all", "gradboost"):
        trained.append(train_gradboost(train_df, val_df, args.save_dir))

    if trained:
        select_best(trained, args.save_dir)

    # Run threshold analysis on best model
    print("\n" + "─" * 57)
    print("  THRESHOLD ANALYSIS")
    print("─" * 57)
    analyse_threshold("gradboost", args.save_dir, val_df)

    print("\n✓ Training complete.")
    print("  Next steps:")
    print("    python nlp/urgency_predictor.py --mode test --model all")
    print("    python evaluation/urgency_eval.py --verbose")
    print("    python nlp/urgency_predictor.py --mode generate")
    print("    jupyter notebook notebooks/04_urgency_prediction.ipynb")


if __name__ == "__main__":
    main()