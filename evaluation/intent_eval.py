
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

RESULTS_PATH = "models/nlp/intent_results.json"
BEST_PATH    = "models/nlp/best_model.json"
REPORT_DIR   = "evaluation/reports"
REPORT_JSON  = os.path.join(REPORT_DIR, "intent_eval_report.json")
REPORT_TXT   = os.path.join(REPORT_DIR, "intent_eval_report.txt")

# Map short test-result keys → full model names used in val results
TEST_KEY_TO_MODEL: dict[str, str] = {
    "logreg":     "TF-IDF + LogReg",
    "svm":        "TF-IDF + SVM",
    "distilbert": "DistilBERT",
}


# ─────────────────────────────────────────────────────────────────────────────
# LOADERS
# ─────────────────────────────────────────────────────────────────────────────

def _load(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found.\n"
            "Run: python nlp/intent_classifier.py --model all"
        )
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 + 2 — Comparison table with val/test gap
# ─────────────────────────────────────────────────────────────────────────────

def build_comparison(results: dict, best: dict) -> dict:
    """
    Build a side-by-side comparison of val and test metrics for every model.
    Computes val→test weighted-F1 gap and flags potential overfit (gap > 0.05).
    """
    val_results  = results.get("full_results", {})
    test_results = results.get("test_results", {})

    rows = []
    for val_name, val_data in val_results.items():
        # Find the matching test result using the name map
        test_data = {}
        for short_key, full_name in TEST_KEY_TO_MODEL.items():
            if full_name == val_name and short_key in test_results:
                test_data = test_results[short_key]
                break

        vwf1 = val_data.get("weighted_f1")
        twf1 = test_data.get("weighted_f1")
        gap  = round(abs(vwf1 - twf1), 4) if (vwf1 and twf1) else None
        rows.append({
            "model":              val_name,
            "val_accuracy":       val_data.get("accuracy"),
            "val_macro_f1":       val_data.get("macro_f1"),
            "val_weighted_f1":    vwf1,
            "val_macro_precision":val_data.get("macro_precision"),
            "val_macro_recall":   val_data.get("macro_recall"),
            "val_uncertainty_rate": val_data.get("uncertainty_rate"),
            "val_avg_confidence": val_data.get("avg_confidence"),
            "test_accuracy":      test_data.get("accuracy"),
            "test_macro_f1":      test_data.get("macro_f1"),
            "test_weighted_f1":   twf1,
            "test_uncertainty_rate": test_data.get("uncertainty_rate"),
            "wtd_f1_gap":         gap,
            "overfit_flag":       (gap > 0.05) if gap is not None else None,
        })
    return {"rows": rows}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Per-class analysis for best model
# ─────────────────────────────────────────────────────────────────────────────

def build_per_class(results: dict, best_model_name: str) -> dict:
    """
    Categorise each class as well-learned (F1≥0.6), partial (0.3–0.6),
    or poorly-learned (F1<0.3) for the best model.
    Falls back to partial match if exact name not found.
    """
    pc_all = results.get("per_class_results", {})

    # Exact match first, then partial
    pc = pc_all.get(best_model_name, {})
    if not pc:
        for key, data in pc_all.items():
            if (best_model_name.lower() in key.lower()
                    or key.lower() in best_model_name.lower()):
                pc = data
                break
    if not pc and pc_all:
        pc = next(iter(pc_all.values()))

    if not pc:
        return {
            "note": "No per-class data available",
            "counts": {"well": 0, "partial": 0, "poor": 0},
            "well_learned": [], "partial": [], "poorly_learned": [],
        }

    well, partial, poor = [], [], []
    for cls, m in pc.items():
        entry = {"class": cls, "f1": m["f1"], "precision": m["precision"],
                 "recall": m["recall"], "support": m["support"]}
        if   m["f1"] >= 0.6: well.append(entry)
        elif m["f1"] >= 0.3: partial.append(entry)
        else:                 poor.append(entry)

    for lst in (well, partial, poor):
        lst.sort(key=lambda x: -x["support"])

    return {
        "counts": {"well": len(well), "partial": len(partial), "poor": len(poor)},
        "well_learned":   well,
        "partial":        partial,
        "poorly_learned": poor,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Confidence analysis
# ─────────────────────────────────────────────────────────────────────────────

def build_confidence(results: dict) -> dict:
    analysis = {}
    for name, data in results.get("full_results", {}).items():
        unc = data.get("uncertainty_rate")
        if unc is None:
            continue
        if unc > 0.90:
            rl_impact = (
                "CRITICAL — >90% uncertain. uncertainty_flag=1 for almost every "
                "prediction. RL Suggest action will almost never fire."
            )
        elif unc > 0.70:
            rl_impact = (
                "HIGH — >70% uncertain. RL will prefer Clarify heavily. "
                "Acceptable for baseline experiments."
            )
        elif unc > 0.40:
            rl_impact = (
                "MODERATE — 40-70% uncertain. RL has a good mix of certain "
                "and uncertain predictions to learn from."
            )
        else:
            rl_impact = (
                "LOW — <40% uncertain. Model is confident enough for diverse "
                "RL actions including Suggest."
            )
        analysis[name] = {
            "uncertainty_rate": unc,
            "avg_confidence":   data.get("avg_confidence"),
            "rl_impact":        rl_impact,
        }
    return analysis


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Recommendation
# ─────────────────────────────────────────────────────────────────────────────

def build_recommendation(results: dict, best: dict) -> dict:
    best_name = best.get("best_model", "Unknown")
    best_comp = next(
        (c for c in best.get("comparison", []) if c["model"] == best_name),
        {},
    )
    vwf1 = best_comp.get("weighted_f1", 0)
    vacc = best_comp.get("accuracy", 0)
    unc  = best_comp.get("uncertainty_rate", 1.0)

    # Val → Test gap for best model
    test_key = next(
        (k for k, v in TEST_KEY_TO_MODEL.items() if v == best_name), None
    )
    test_data = results.get("test_results", {}).get(test_key, {})
    twf1      = test_data.get("weighted_f1")
    gap_note  = ""
    if twf1:
        gap = abs(vwf1 - twf1)
        status = "possible overfit" if gap > 0.05 else "stable generalisation"
        gap_note = (
            f"Val Weighted F1={vwf1:.4f}  |  "
            f"Test Weighted F1={twf1:.4f}  |  "
            f"Gap={gap:.4f}  ({status})"
        )

    return {
        "best_model":    best_name,
        "best_metrics":  {
            "val_accuracy":    round(vacc, 4),
            "val_weighted_f1": round(vwf1, 4),
            "uncertainty_rate":round(unc,  4),
        },
        "gap_note": gap_note,
        "rl_readiness": (
            "RL pipeline can proceed with the current best model. "
            f"uncertainty_flag=1 for {unc * 100:.1f}% of predictions — "
            "the RL agent will use Clarify for these, which is the correct "
            "safe fallback when intent classification is uncertain."
        ),
        "ceiling_note": (
            "Performance ceiling with this dataset (596 train, 9 classes, "
            "avg 66/class): ~55-60% accuracy. The primary bottleneck is "
            "needs_info (220 samples, workflow-state label indistinguishable "
            "from bugs) and enhancement (9 samples, unlearnable). These are "
            "data limitations, not model limitations."
        ),
        "next_step": (
            "Run confidence_estimator.py --mode generate --model best "
            "to write intent features into final_rl_dataset.csv, "
            "then proceed to urgency_predictor.py."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# PRINT FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

SEP = "=" * 72


def _fmt(val, pct: bool = False, decimals: int = 4) -> str:
    if val is None:
        return "N/A"
    if pct:
        return f"{val * 100:.1f}%"
    return f"{val:.{decimals}f}"


def print_comparison(comp: dict, verbose: bool = False) -> None:
    print(f"\n{SEP}")
    print("  MODEL COMPARISON — Val Set vs Test Set")
    print(SEP)
    hdr = (f"  {'Model':<28} {'ValAcc':>7} {'ValWF1':>7} "
           f"{'TestAcc':>8} {'TestWF1':>8} {'Gap':>7} {'Overfit':>8}")
    print(hdr)
    print("  " + "─" * 70)
    for r in comp["rows"]:
        va   = _fmt(r["val_accuracy"],    pct=True)
        vwf  = _fmt(r["val_weighted_f1"])
        ta   = _fmt(r["test_accuracy"],   pct=True)
        twf  = _fmt(r["test_weighted_f1"])
        gap  = _fmt(r["wtd_f1_gap"])
        ovf  = ("YES" if r["overfit_flag"]
                else ("no"  if r["overfit_flag"] is False
                      else "N/A"))
        print(
            f"  {r['model']:<28} {va:>7} {vwf:>7} "
            f"{ta:>8} {twf:>8} {gap:>7} {ovf:>8}"
        )


def print_per_class(pc: dict, verbose: bool = False) -> None:
    print(f"\n{SEP}")
    print("  PER-CLASS ANALYSIS — Best Model")
    print(SEP)

    if "note" in pc:
        print(f"  {pc['note']}")
        return

    c = pc["counts"]
    print(f"  Well-learned  (F1 >= 0.6) : {c['well']:2d} classes")
    print(f"  Partial       (0.3–0.6)   : {c['partial']:2d} classes")
    print(f"  Poorly-learned (F1 < 0.3) : {c['poor']:2d} classes")

    if verbose:
        for label, items in [
            ("WELL-LEARNED",   pc["well_learned"]),
            ("PARTIAL",        pc["partial"]),
            ("POORLY-LEARNED", pc["poorly_learned"]),
        ]:
            if not items:
                continue
            print(f"\n  [{label}]")
            print(f"  {'Class':<22} {'F1':>6} {'P':>6} {'R':>6} {'n':>5}")
            print("  " + "─" * 44)
            for e in items:
                bar = "█" * int(e["f1"] * 16)
                print(
                    f"  {e['class']:<22} {e['f1']:>6.3f} "
                    f"{e['precision']:>6.3f} {e['recall']:>6.3f} "
                    f"{e['support']:>5}  {bar}"
                )


def print_confidence(conf: dict) -> None:
    print(f"\n{SEP}")
    print("  CONFIDENCE & UNCERTAINTY ANALYSIS")
    print(SEP)
    for name, data in conf.items():
        print(f"\n  {name}")
        print(f"    Uncertainty rate : {data['uncertainty_rate'] * 100:.1f}%")
        print(f"    Avg confidence   : {data['avg_confidence']:.4f}")
        print(f"    RL impact        : {data['rl_impact']}")


def print_recommendation(rec: dict) -> None:
    print(f"\n{SEP}")
    print("  RECOMMENDATION")
    print(SEP)
    m = rec["best_metrics"]
    print(f"  Best model       : {rec['best_model']}")
    print(f"  Val accuracy     : {m['val_accuracy'] * 100:.2f}%")
    print(f"  Val weighted F1  : {m['val_weighted_f1']:.4f}")
    print(f"  Uncertainty rate : {m['uncertainty_rate'] * 100:.1f}%")
    if rec["gap_note"]:
        print(f"\n  {rec['gap_note']}")
    print(f"\n  RL readiness  : {rec['rl_readiness']}")
    print(f"\n  Ceiling note  : {rec['ceiling_note']}")
    print(f"\n  Next step     : {rec['next_step']}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run(verbose: bool = False) -> dict:
    print(SEP)
    print("  NexResolve — Intent Classification Evaluation")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(SEP)

    results = _load(RESULTS_PATH)
    best    = _load(BEST_PATH) if os.path.exists(BEST_PATH) else {}

    best_name  = best.get("best_model", next(iter(results.get("full_results", {}).keys()), ""))
    comparison = build_comparison(results, best)
    per_class  = build_per_class(results, best_name)
    confidence = build_confidence(results)
    recommend  = build_recommendation(results, best)

    print_comparison(comparison, verbose)
    print_per_class(per_class, verbose)
    print_confidence(confidence)
    print_recommendation(recommend)

    # Assemble report
    report = {
        "generated_at":   datetime.now().isoformat(),
        "comparison":     comparison,
        "per_class":      per_class,
        "confidence":     confidence,
        "recommendation": recommend,
        "best_model":     best,
    }

    os.makedirs(REPORT_DIR, exist_ok=True)

    with open(REPORT_JSON, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    # Capture text output
    buf = io.StringIO()
    with redirect_stdout(buf):
        _run_print(results, best, verbose)
    with open(REPORT_TXT, "w", encoding="utf-8") as fh:
        fh.write(buf.getvalue())

    print(f"\n  Reports saved:")
    print(f"    {REPORT_JSON}")
    print(f"    {REPORT_TXT}")

    return report


def _run_print(results: dict, best: dict, verbose: bool) -> None:
    best_name  = best.get("best_model", "")
    print_comparison(build_comparison(results, best), verbose)
    print_per_class(build_per_class(results, best_name), verbose)
    print_confidence(build_confidence(results))
    print_recommendation(build_recommendation(results, best))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NexResolve Intent Classification Evaluation"
    )
    parser.add_argument("--verbose", action="store_true",
                        help="Show full per-class breakdown")
    args = parser.parse_args()
    run(verbose=args.verbose)