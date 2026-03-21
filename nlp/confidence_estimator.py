

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
# THRESHOLDS
# ─────────────────────────────────────────────────────────────────────────────

UNCERTAINTY_THRESHOLD: float = 0.60   # primary flag cutoff
MEDIUM_THRESHOLD:      float = 0.80   # above this → high confidence

RL_RECOMMENDATIONS: dict[str, str] = {
    "low":    "clarify_first",     # confidence < 0.60
    "medium": "route_or_clarify",  # 0.60 ≤ confidence < 0.80
    "high":   "suggest_or_route",  # confidence ≥ 0.80
}


# ─────────────────────────────────────────────────────────────────────────────
# CORE — single prediction
# ─────────────────────────────────────────────────────────────────────────────

def estimate(
    confidence_score: float,
    threshold: float = UNCERTAINTY_THRESHOLD,
) -> dict:
    """
    Map a confidence score to uncertainty flag, band, and RL recommendation.

    Args:
        confidence_score: max predicted class probability from any classifier
        threshold:        cutoff for uncertainty flag (default 0.60)

    Returns:
        dict with uncertainty_flag, confidence_band, rl_recommendation,
        threshold_used
    """
    if not 0.0 <= confidence_score <= 1.0:
        raise ValueError(
            f"confidence_score must be in [0, 1], got {confidence_score}"
        )

    if confidence_score < UNCERTAINTY_THRESHOLD:
        band = "low"
    elif confidence_score < MEDIUM_THRESHOLD:
        band = "medium"
    else:
        band = "high"

    return {
        "uncertainty_flag":  int(confidence_score < threshold),
        "confidence_band":   band,
        "rl_recommendation": RL_RECOMMENDATIONS[band],
        "threshold_used":    threshold,
    }


def estimate_batch(
    confidence_scores: np.ndarray,
    threshold: float = UNCERTAINTY_THRESHOLD,
) -> pd.DataFrame:
    """
    Run estimate() over an array of confidence scores.
    Returns a DataFrame with one row per issue.
    """
    rows = [
        {"confidence_score": round(float(s), 4), **estimate(float(s), threshold)}
        for s in confidence_scores
    ]
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# GENERATE FEATURES — writes to data/final CSVs
# ─────────────────────────────────────────────────────────────────────────────

def generate_features(
    model_type: str = "best",
    save_dir: str   = "models/nlp",
    issues_path: str = "data/final/cleaned_issues.csv",
    rl_path: str     = "data/final/final_rl_dataset.csv",
    threshold: float = UNCERTAINTY_THRESHOLD,
) -> pd.DataFrame:
    """
    Run the best trained classifier on all issues and write:
      confidence_score, uncertainty_flag, confidence_band,
      rl_recommendation, suggested_action, intent_group
    into cleaned_issues.csv and final_rl_dataset.csv.

    This makes the RL state vector complete.  Run this AFTER the best
    model has been selected (i.e. after all three models are trained and
    intent_eval.py has been run).

    Args:
        model_type: 'best' | 'logreg' | 'svm' | 'distilbert'
        save_dir:   folder containing saved models
        issues_path: path to cleaned_issues.csv
        rl_path:     path to final_rl_dataset.csv
        threshold:   uncertainty cutoff (default 0.60)
    """
    from nlp.intent_classifier import INTENT_TO_ACTION, map_labels

    # ── Resolve 'best' to actual model type ──────────────────────────────────
    if model_type == "best":
        best_path = os.path.join(save_dir, "best_model.json")
        if not os.path.exists(best_path):
            raise FileNotFoundError(
                f"No best_model.json at {best_path}. "
                "Train models first, then run intent_eval.py."
            )
        with open(best_path, encoding="utf-8") as fh:
            best = json.load(fh)
        mname = best["best_model"]
        if "DistilBERT" in mname:
            model_type = "distilbert"
        elif "SVM" in mname:
            model_type = "svm"
        else:
            model_type = "logreg"
        print(f"[confidence_estimator] Best model: {mname} → using {model_type}")

    print(f"[confidence_estimator] Running {model_type} on all issues ...")

    ci = pd.read_csv(issues_path, encoding="utf-8")
    texts = ci["clean_text"].fillna("").astype(str).tolist()
    print(f"  Issues to process: {len(texts)}")

    # ── Run model to get probabilities ───────────────────────────────────────
    if model_type in ("logreg", "svm"):
        pkl = os.path.join(
            save_dir,
            "intent_tfidf_logreg.pkl" if model_type == "logreg"
            else "intent_tfidf_svm.pkl",
        )
        if not os.path.exists(pkl):
            raise FileNotFoundError(
                f"Model not found: {pkl}\n"
                f"Run: python nlp/intent_classifier.py --model {model_type}"
            )
        with open(pkl, "rb") as fh:
            obj = pickle.load(fh)
        pipeline = obj["pipeline"]
        probs    = pipeline.predict_proba(texts)
        preds    = pipeline.predict(texts)
        confs    = probs.max(axis=1)

    elif model_type == "distilbert":
        try:
            import torch
            from transformers import (
                DistilBertForSequenceClassification,
                DistilBertTokenizerFast,
            )
        except ImportError as exc:
            raise ImportError("pip install transformers torch") from exc

        ckpt = os.path.join(save_dir, "intent_distilbert")
        ckpt = ckpt.replace("\\", "/")
        if not os.path.exists(ckpt):
            raise FileNotFoundError(
                f"DistilBERT checkpoint not found: {ckpt}\n"
                "Run: python nlp/intent_classifier.py --model distilbert"
            )

        device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = DistilBertTokenizerFast.from_pretrained(ckpt)
        model     = DistilBertForSequenceClassification.from_pretrained(ckpt).to(device)
        model.eval()
        with open(os.path.join(ckpt, "label_encoder.pkl"), "rb") as fh:
            le = pickle.load(fh)

        all_preds, all_confs = [], []
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            enc   = tokenizer(
                batch, max_length=128, truncation=True,
                padding=True, return_tensors="pt",
            )
            with torch.no_grad():
                out   = model(
                    input_ids=enc["input_ids"].to(device),
                    attention_mask=enc["attention_mask"].to(device),
                )
                probs_b = torch.softmax(out.logits, dim=1).cpu().numpy()
            all_preds.extend(le.inverse_transform(probs_b.argmax(axis=1)).tolist())
            all_confs.extend(probs_b.max(axis=1).tolist())

        preds = all_preds
        confs = np.array(all_confs)

    else:
        raise ValueError(
            f"Unknown model_type: '{model_type}'. "
            "Choose: best | logreg | svm | distilbert"
        )

    # ── Build feature DataFrame ───────────────────────────────────────────────
    features = estimate_batch(confs, threshold)
    features["intent_group"]     = preds
    features["issue_number"]     = ci["issue_number"].values
    features["suggested_action"] = [
        INTENT_TO_ACTION.get(g, "clarify") for g in preds
    ]

    # ── Write to cleaned_issues.csv ───────────────────────────────────────────
    drop_cols = [
        "intent_group", "confidence_score", "uncertainty_flag",
        "confidence_band", "rl_recommendation", "suggested_action",
    ]
    for col in drop_cols:
        if col in ci.columns:
            ci = ci.drop(columns=[col])

    for col in drop_cols:
        ci[col] = features[col].values

    ci.to_csv(issues_path, index=False, encoding="utf-8")
    print(f"  Updated → {issues_path}")

    # ── Write to final_rl_dataset.csv ─────────────────────────────────────────
    rl = pd.read_csv(rl_path, encoding="utf-8")
    for col in drop_cols:
        if col in rl.columns:
            rl = rl.drop(columns=[col])

    merge_cols = ["issue_number"] + drop_cols
    rl = rl.merge(features[merge_cols], on="issue_number", how="left")
    rl.to_csv(rl_path, index=False, encoding="utf-8")
    print(f"  Updated → {rl_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    unc_pct    = features["uncertainty_flag"].mean() * 100
    band_counts = features["confidence_band"].value_counts().to_dict()
    print(f"\n  Total processed:  {len(features)}")
    print(f"  Uncertain (flag=1): {int(features['uncertainty_flag'].sum())}  ({unc_pct:.1f}%)")
    print(f"  Confidence bands:   {band_counts}")
    print(f"  Intent distribution:")
    for intent, count in features["intent_group"].value_counts().items():
        print(f"    {intent:20s}: {count}")

    return features


# ─────────────────────────────────────────────────────────────────────────────
# THRESHOLD ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def analyse_threshold(
    model_type: str = "svm",
    save_dir: str   = "models/nlp",
    val_path: str   = "data/splits/val.csv",
    thresholds: Optional[list] = None,
) -> dict:
    """
    Evaluate uncertainty threshold choices on the val set.
    For each threshold:
      certain_acc   — accuracy when model is confident  (flag=0)
      uncertain_acc — accuracy when model is uncertain  (flag=1)
      signal        — certain_acc − uncertain_acc

    A good threshold maximises signal: certain predictions are much more
    accurate than uncertain ones.  Saves to models/nlp/threshold_analysis.json.
    """
    from nlp.intent_classifier import map_labels

    if thresholds is None:
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    val_df = pd.read_csv(val_path, encoding="utf-8")
    y_true = map_labels(val_df["primary_label"])

    # Get probabilities from model
    if model_type in ("logreg", "svm"):
        pkl = os.path.join(
            save_dir,
            "intent_tfidf_logreg.pkl" if model_type == "logreg"
            else "intent_tfidf_svm.pkl",
        )
        with open(pkl, "rb") as fh:
            obj = pickle.load(fh)
        probs = obj["pipeline"].predict_proba(val_df["clean_text"].fillna(""))
        preds = obj["pipeline"].predict(val_df["clean_text"].fillna(""))
        confs = probs.max(axis=1)

    elif model_type == "distilbert":
        import torch
        from transformers import (
            DistilBertForSequenceClassification,
            DistilBertTokenizerFast,
        )
        ckpt    = os.path.join(save_dir, "intent_distilbert").replace("\\", "/")
        device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tok     = DistilBertTokenizerFast.from_pretrained(ckpt)
        model   = DistilBertForSequenceClassification.from_pretrained(ckpt).to(device)
        model.eval()
        with open(os.path.join(ckpt, "label_encoder.pkl"), "rb") as fh:
            le = pickle.load(fh)
        texts = val_df["clean_text"].fillna("").astype(str).tolist()
        all_p, all_c = [], []
        for i in range(0, len(texts), 32):
            batch = texts[i: i + 32]
            enc   = tok(batch, max_length=128, truncation=True,
                        padding=True, return_tensors="pt")
            with torch.no_grad():
                out   = model(input_ids=enc["input_ids"].to(device),
                              attention_mask=enc["attention_mask"].to(device))
                pb    = torch.softmax(out.logits, dim=1).cpu().numpy()
            all_p.extend(le.inverse_transform(pb.argmax(axis=1)).tolist())
            all_c.extend(pb.max(axis=1).tolist())
        preds = np.array(all_p)
        confs = np.array(all_c)

    elif model_type == "best":
        best_path = os.path.join(save_dir, "best_model.json")
        with open(best_path, encoding="utf-8") as fh:
            best = json.load(fh)
        mname = best["best_model"]
        mt = "distilbert" if "DistilBERT" in mname else ("svm" if "SVM" in mname else "logreg")
        return analyse_threshold(mt, save_dir, val_path, thresholds)

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    correct = (preds == y_true.values)

    results = {}
    print(f"\n[threshold_analysis]  model={model_type}")
    print(f"  {'Threshold':>10} | {'Unc%':>7} | {'CertainAcc':>11} | "
          f"{'UncertainAcc':>13} | {'Signal':>8}")
    print("  " + "─" * 58)

    for t in thresholds:
        unc_mask  = confs < t
        n_unc     = unc_mask.sum()
        n_cer     = (~unc_mask).sum()
        acc_cer   = float(correct[~unc_mask].mean()) if n_cer   > 0 else 0.0
        acc_unc   = float(correct[unc_mask].mean())  if n_unc   > 0 else 0.0
        unc_pct   = n_unc / len(confs) * 100
        signal    = acc_cer - acc_unc

        results[str(t)] = {
            "threshold":     t,
            "uncertain_pct": round(unc_pct,  2),
            "certain_acc":   round(acc_cer,  4),
            "uncertain_acc": round(acc_unc,  4),
            "signal":        round(signal,   4),
            "n_uncertain":   int(n_unc),
            "n_certain":     int(n_cer),
        }
        print(
            f"  {t:>10.1f} | {unc_pct:>6.1f}% | "
            f"{acc_cer * 100:>10.1f}% | "
            f"{acc_unc * 100:>12.1f}% | "
            f"{signal:>8.4f}"
        )

    best_t = max(results, key=lambda k: results[k]["signal"])
    print(f"\n  Recommended threshold: {best_t}  "
          f"(signal={results[best_t]['signal']:.4f})")

    out = {
        "model_type":             model_type,
        "per_threshold":          results,
        "recommended_threshold":  float(best_t),
    }
    out_path = os.path.join(save_dir, "threshold_analysis.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"  Saved → {out_path}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="NexResolve Confidence Estimator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode", choices=["generate", "analyse", "estimate"],
        default="generate",
    )
    parser.add_argument(
        "--model", default="best",
        choices=["best", "logreg", "svm", "distilbert"],
    )
    parser.add_argument("--score",     type=float, default=None,
                        help="Confidence score for --mode estimate")
    parser.add_argument("--threshold", type=float,
                        default=UNCERTAINTY_THRESHOLD)
    parser.add_argument("--save_dir",  default="models/nlp")
    args = parser.parse_args()

    if args.mode == "estimate":
        score  = args.score if args.score is not None else float(
            input("Enter confidence score (0.0–1.0): ")
        )
        result = estimate(score, args.threshold)
        print(json.dumps(result, indent=2))

    elif args.mode == "generate":
        generate_features(
            model_type=args.model,
            save_dir=args.save_dir,
            threshold=args.threshold,
        )
        print("\n✓ Done. Confidence features written to CSVs.")

    elif args.mode == "analyse":
        analyse_threshold(
            model_type=args.model,
            save_dir=args.save_dir,
        )


if __name__ == "__main__":
    main()