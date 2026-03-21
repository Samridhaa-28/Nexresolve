
from __future__ import annotations

import argparse
import io
import json
import os
import sys
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RESULTS_PATH = "models/nlp/urgency_results.json"
BEST_PATH    = "models/nlp/urgency_best_model.json"
TA_PATH      = "models/nlp/urgency_threshold_analysis.json"
REPORT_DIR   = "evaluation/reports"
REPORT_JSON  = os.path.join(REPORT_DIR, "urgency_eval_report.json")
REPORT_TXT   = os.path.join(REPORT_DIR, "urgency_eval_report.txt")

SEP = "=" * 72


def _load(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found.\n"
            "Run: python nlp/urgency_predictor.py --model all"
        )
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _fmt(val, pct: bool = False) -> str:
    if val is None: return "N/A"
    return f"{val*100:.2f}%" if pct else f"{val:.4f}"


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Model Comparison
# ─────────────────────────────────────────────────────────────────────────────

# Map full model names → short test keys
VAL_TO_TEST = {
    "Rule-Based":          "rule_based",
    "Logistic Regression": "logreg",
    "Gradient Boosting":   "gradboost",
}

def build_comparison(results: dict, best: dict) -> dict:
    val_results  = results.get("full_results", {})
    test_results = results.get("test_results", {})

    rows = []
    for name, vdata in val_results.items():
        test_key = VAL_TO_TEST.get(name, name.lower().replace(" ", "_"))
        tdata = test_results.get(test_key, test_results.get(name, {}))
        vf1   = vdata.get("f1")
        tf1   = tdata.get("f1")
        gap   = round(abs(vf1 - tf1), 4) if (vf1 and tf1) else None

        rows.append({
            "model":          name,
            "val_accuracy":   vdata.get("accuracy"),
            "val_precision":  vdata.get("precision"),
            "val_recall":     vdata.get("recall"),
            "val_f1":         vf1,
            "val_roc_auc":    vdata.get("roc_auc"),
            "val_avg_prec":   vdata.get("avg_precision"),
            "test_accuracy":  tdata.get("accuracy"),
            "test_f1":        tf1,
            "test_roc_auc":   tdata.get("roc_auc"),
            "f1_gap":         gap,
            "overfit":        (gap > 0.05) if gap is not None else None,
        })
    return {"rows": rows}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Threshold Analysis Summary
# ─────────────────────────────────────────────────────────────────────────────

def build_threshold_summary() -> dict:
    if not os.path.exists(TA_PATH):
        return {"note": "Threshold analysis not run yet."}
    with open(TA_PATH, encoding="utf-8") as fh:
        ta = json.load(fh)
    return {
        "model_type":            ta.get("model_type"),
        "recommended_threshold": ta.get("recommended_threshold"),
        "recommended_f1":        ta.get("recommended_f1"),
        "selection_criterion":   ta.get("selection_criterion"),
        "per_threshold":         ta.get("per_threshold", {}),
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Feature Importance
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_analysis(results: dict) -> dict:
    full = results.get("full_results", {})
    out  = {}
    for name, data in full.items():
        if "feature_importances" in data:
            out[name] = {"type": "importance", "values": data["feature_importances"]}
        elif "feature_coefficients" in data:
            out[name] = {"type": "coefficient", "values": data["feature_coefficients"]}
    return out


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Recommendation
# ─────────────────────────────────────────────────────────────────────────────

def build_recommendation(results: dict, best: dict, ta: dict) -> dict:
    best_name = best.get("best_model", "Unknown")
    best_comp = next(
        (c for c in best.get("comparison", []) if c["model"] == best_name), {}
    )
    test_data = results.get("test_results", {}).get(best_name, {})

    vf1 = best_comp.get("f1", 0)
    tf1 = test_data.get("f1")
    gap_note = ""
    if tf1:
        gap    = abs(vf1 - tf1)
        status = "possible overfit" if gap > 0.05 else "stable"
        gap_note = f"Val F1={vf1:.4f}  |  Test F1={tf1:.4f}  |  Gap={gap:.4f}  ({status})"

    thresh = ta.get("recommended_threshold", 0.5)
    return {
        "best_model":       best_name,
        "val_f1":           round(vf1, 4),
        "val_recall":       round(best_comp.get("recall", 0), 4),
        "val_roc_auc":      round(best_comp.get("roc_auc", 0), 4),
        "threshold":        thresh,
        "gap_note":         gap_note,
        "rl_readiness": (
            f"urgent_flag is set when urgency_score >= {thresh}. "
            "The RL agent uses this flag to prioritise tickets and weight rewards. "
            "High urgency + SLA breach penalty drives the agent to resolve "
            "urgent tickets faster."
        ),
        "next_step": (
            "Run urgency_predictor.py --mode generate to write urgency_score "
            "and urgent_flag into final_rl_dataset.csv, "
            "then proceed to entity_extractor.py."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# PRINT FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def print_comparison(comp: dict, verbose: bool = False) -> None:
    print(f"\n{SEP}")
    print("  MODEL COMPARISON — Val Set vs Test Set")
    print(SEP)
    print(f"  {'Model':<25} {'ValAcc':>7} {'ValPrec':>8} {'ValRec':>7} "
          f"{'ValF1':>7} {'ValAUC':>7} {'TestF1':>8} {'Gap':>7} {'OF':>5}")
    print("  " + "─" * 72)

    for r in comp["rows"]:
        of = ("YES" if r["overfit"] else ("no" if r["overfit"] is False else "N/A"))
        print(
            f"  {r['model']:<25}"
            f"{_fmt(r['val_accuracy'], pct=True):>7}"
            f"{_fmt(r['val_precision']):>9}"
            f"{_fmt(r['val_recall']):>8}"
            f"{_fmt(r['val_f1']):>8}"
            f"{_fmt(r['val_roc_auc']):>8}"
            f"{_fmt(r['test_f1']):>9}"
            f"{_fmt(r['f1_gap']):>8}"
            f"{of:>6}"
        )


def print_threshold(ta: dict) -> None:
    print(f"\n{SEP}")
    print("  THRESHOLD ANALYSIS")
    print(SEP)
    if "note" in ta:
        print(f"  {ta['note']}")
        return
    print(f"  Model:                 {ta.get('model_type')}")
    print(f"  Selection criterion:   {ta.get('selection_criterion')}")
    print(f"  Recommended threshold: {ta.get('recommended_threshold')}")
    print(f"  F1 at recommended:     {ta.get('recommended_f1')}")
    print()
    pt = ta.get("per_threshold", {})
    if pt:
        print(f"  {'T':>6} | {'Prec':>7} {'Rec':>7} {'F1':>7} {'Acc':>7} | {'Urgent%':>8}")
        print("  " + "─" * 52)
        best_t = str(ta.get("recommended_threshold"))
        for t_str, vals in pt.items():
            marker = " ←" if t_str == best_t else ""
            print(
                f"  {vals['threshold']:>6.2f} | "
                f"{vals['precision']:>7.3f} "
                f"{vals['recall']:>7.3f} "
                f"{vals['f1']:>7.3f} "
                f"{vals['accuracy']:>7.3f} | "
                f"{vals['urgent_pct']:>7.1f}%{marker}"
            )


def print_features(feat: dict, verbose: bool = False) -> None:
    if not feat or not verbose:
        return
    print(f"\n{SEP}")
    print("  FEATURE ANALYSIS")
    print(SEP)
    for model, data in feat.items():
        print(f"\n  {model} — {data['type']}:")
        vals = data["values"]
        for fname, val in sorted(vals.items(), key=lambda x: -abs(x[1])):
            bar = "█" * int(abs(val) * 30)
            sign = "+" if val > 0 else ""
            print(f"    {fname:30s}: {sign}{val:>8.4f}  |{bar}")


def print_recommendation(rec: dict) -> None:
    print(f"\n{SEP}")
    print("  RECOMMENDATION")
    print(SEP)
    print(f"  Best model     : {rec['best_model']}")
    print(f"  Val F1         : {rec['val_f1']:.4f}  ← selection metric")
    print(f"  Val Recall     : {rec['val_recall']:.4f}  (catching urgent tickets)")
    print(f"  Val ROC-AUC    : {rec['val_roc_auc']:.4f}")
    print(f"  Threshold      : {rec['threshold']}")
    if rec["gap_note"]:
        print(f"\n  {rec['gap_note']}")
    print(f"\n  RL readiness : {rec['rl_readiness']}")
    print(f"\n  Next step    : {rec['next_step']}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run(verbose: bool = False) -> dict:
    print(SEP)
    print("  NexResolve — Urgency Prediction Evaluation")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(SEP)

    results    = _load(RESULTS_PATH)
    best       = _load(BEST_PATH) if os.path.exists(BEST_PATH) else {}
    ta         = build_threshold_summary()
    comparison = build_comparison(results, best)
    feat       = build_feature_analysis(results)
    rec        = build_recommendation(results, best, ta)

    print_comparison(comparison, verbose)
    print_threshold(ta)
    print_features(feat, verbose)
    print_recommendation(rec)

    report = {
        "generated_at":   datetime.now().isoformat(),
        "comparison":     comparison,
        "threshold":      ta,
        "feature_analysis": feat,
        "recommendation": rec,
        "best_model":     best,
    }

    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(REPORT_JSON, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    buf = io.StringIO()
    with redirect_stdout(buf):
        _run_print(results, best, ta, verbose)
    with open(REPORT_TXT, "w", encoding="utf-8") as fh:
        fh.write(buf.getvalue())

    print(f"\n  Reports saved:")
    print(f"    {REPORT_JSON}")
    print(f"    {REPORT_TXT}")
    return report


def _run_print(results, best, ta, verbose):
    comparison = build_comparison(results, best)
    feat       = build_feature_analysis(results)
    rec        = build_recommendation(results, best, ta)
    print_comparison(comparison, verbose)
    print_threshold(ta)
    print_features(feat, verbose)
    print_recommendation(rec)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NexResolve Urgency Evaluation"
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    run(verbose=args.verbose)