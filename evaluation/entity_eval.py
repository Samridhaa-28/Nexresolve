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

RESULTS_PATH = "models/nlp/entity_results.json"
BEST_PATH    = "models/nlp/entity_best_model.json"
REPORT_DIR   = "evaluation/reports"
REPORT_JSON  = os.path.join(REPORT_DIR, "entity_eval_report.json")
REPORT_TXT   = os.path.join(REPORT_DIR, "entity_eval_report.txt")

SEP  = "=" * 72
ENTITY_TYPES = ["version", "error_type", "platform", "hardware"]


def _load(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found.\n"
            "Run: python nlp/entity_extractor.py --mode train"
        )
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Approach comparison
# ─────────────────────────────────────────────────────────────────────────────

def build_comparison(results: dict, best: dict) -> dict:
    val_results      = results.get("val_results", {})
    # test_results_all has all approaches; test_results has best only
    test_results_all = results.get("test_results_all", {})
    test_results_best= results.get("test_results", {})

    rows = []
    for name, vdata in val_results.items():
        # Try test_results_all first, fall back to test_results (best only)
        tdata = test_results_all.get(name, {})
        if not tdata:
            best_clean = test_results_best.get("approach_name_clean", "")
            if best_clean == name:
                tdata = test_results_best

        vf1 = vdata.get("macro_f1")
        tf1 = tdata.get("macro_f1")
        gap = round(abs(vf1 - tf1), 4) if (vf1 and tf1) else None

        rows.append({
            "approach":        name,
            "val_macro_p":     vdata.get("macro_precision"),
            "val_macro_r":     vdata.get("macro_recall"),
            "val_macro_f1":    vf1,
            "test_macro_f1":   tf1,
            "f1_gap":          gap,
            "overfit":         (gap > 0.05) if gap is not None else None,
        })
    return {"rows": rows}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Per-entity breakdown
# ─────────────────────────────────────────────────────────────────────────────

def build_per_entity(results: dict, best_name: str) -> dict:
    val_results = results.get("val_results", {})
    best_data   = val_results.get(best_name, {})
    per_entity  = best_data.get("per_entity", {})

    rows = []
    for etype in ENTITY_TYPES:
        m = per_entity.get(etype, {})
        rows.append({
            "entity_type": etype,
            "precision":   m.get("precision"),
            "recall":      m.get("recall"),
            "f1":          m.get("f1"),
            "support":     m.get("support"),
            "tp":          m.get("tp"),
            "fp":          m.get("fp"),
            "fn":          m.get("fn"),
        })
    return {"rows": rows, "best_approach": best_name}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Annotation strategy note
# ─────────────────────────────────────────────────────────────────────────────

def build_annotation_note(best: dict) -> str:
    return best.get("annotation_strategy", "Hybrid Union Silver Standard")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Recommendation
# ─────────────────────────────────────────────────────────────────────────────

def build_recommendation(results: dict, best: dict) -> dict:
    best_name = best.get("best_approach", "Unknown")
    best_comp = next(
        (c for c in best.get("comparison", []) if c["approach"] == best_name), {}
    )
    test_data = results.get("test_results", {})
    vf1 = best_comp.get("macro_f1", 0)
    tf1 = test_data.get("macro_f1")
    gap_note = ""
    if tf1:
        gap    = abs(vf1 - tf1)
        status = "possible overfit" if gap > 0.05 else "stable"
        gap_note = (
            f"Val Macro F1={vf1:.4f}  |  "
            f"Test Macro F1={tf1:.4f}  |  "
            f"Gap={gap:.4f}  ({status})"
        )

    return {
        "best_approach":   best_name,
        "val_macro_f1":    round(vf1, 4),
        "selection_metric":best.get("selection_metric"),
        "selection_reason":best.get("selection_reason"),
        "gap_note":        gap_note,
        "annotation":      best.get("annotation_strategy"),
        "rl_readiness": (
            "Entity flags (has_version, has_error_type, has_platform, has_hardware) "
            "are now in final_rl_dataset.csv. The RL agent uses these to understand "
            "how complete the ticket information is. entity_count=0 → likely needs "
            "clarification. entity_count=4 → fully specified → safe to route."
        ),
        "next_step": (
            "Run entity_extractor.py --mode generate to write flags into CSVs, "
            "then proceed to missing_detector.py (Step 8)."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# PRINT FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(val, pct: bool = False) -> str:
    if val is None: return "N/A"
    return f"{val*100:.2f}%" if pct else f"{val:.4f}"


def print_comparison(comp: dict) -> None:
    print(f"\n{SEP}")
    print("  APPROACH COMPARISON  (Val vs Test)")
    print(SEP)
    print(f"  {'Approach':<12} {'MacroP':>9} {'MacroR':>9} {'ValF1':>9} "
          f"{'TestF1':>9} {'Gap':>8} {'OF':>5}")
    print("  " + "─" * 60)
    for r in comp["rows"]:
        of = ("YES" if r["overfit"] else ("no" if r["overfit"] is False else "N/A"))
        print(
            f"  {r['approach']:<12}"
            f"{_fmt(r['val_macro_p']):>10}"
            f"{_fmt(r['val_macro_r']):>10}"
            f"{_fmt(r['val_macro_f1']):>10}"
            f"{_fmt(r['test_macro_f1']):>10}"
            f"{_fmt(r['f1_gap']):>9}"
            f"{of:>6}"
        )


def print_per_entity(pe: dict, verbose: bool = False) -> None:
    print(f"\n{SEP}")
    print(f"  PER-ENTITY BREAKDOWN — {pe['best_approach']}")
    print(SEP)
    print(f"  {'Entity':15s} {'Prec':>7} {'Rec':>7} {'F1':>7} "
          f"{'Support':>8} {'TP':>5} {'FP':>5} {'FN':>5}")
    print("  " + "─" * 60)
    for r in pe["rows"]:
        bar = "█" * int((r["f1"] or 0) * 20)
        print(
            f"  {r['entity_type']:15s}"
            f"{_fmt(r['precision']):>8}"
            f"{_fmt(r['recall']):>8}"
            f"{_fmt(r['f1']):>8}"
            f"{str(r['support'] or 'N/A'):>9}"
            f"{str(r['tp'] or 0):>6}"
            f"{str(r['fp'] or 0):>6}"
            f"{str(r['fn'] or 0):>6}"
            f"  |{bar}"
        )


def print_annotation(note: str) -> None:
    print(f"\n{SEP}")
    print("  ANNOTATION STRATEGY")
    print(SEP)
    print(f"  {note}")


def print_recommendation(rec: dict) -> None:
    print(f"\n{SEP}")
    print("  RECOMMENDATION")
    print(SEP)
    print(f"  Best approach    : {rec['best_approach']}")
    print(f"  Val Macro F1     : {rec['val_macro_f1']:.4f}  ← selection metric")
    print(f"  Selection reason : {rec['selection_reason']}")
    if rec["gap_note"]:
        print(f"\n  {rec['gap_note']}")
    print(f"\n  RL readiness : {rec['rl_readiness']}")
    print(f"\n  Next step    : {rec['next_step']}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run(verbose: bool = False) -> dict:
    print(SEP)
    print("  NexResolve — Entity Extraction Evaluation")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(SEP)

    results    = _load(RESULTS_PATH)
    best       = _load(BEST_PATH) if os.path.exists(BEST_PATH) else {}
    best_name  = best.get("best_approach", "")
    comparison = build_comparison(results, best)
    per_entity = build_per_entity(results, best_name)
    annot_note = build_annotation_note(best)
    rec        = build_recommendation(results, best)

    print_comparison(comparison)
    print_per_entity(per_entity, verbose)
    print_annotation(annot_note)
    print_recommendation(rec)

    report = {
        "generated_at": datetime.now().isoformat(),
        "comparison":   comparison,
        "per_entity":   per_entity,
        "annotation":   annot_note,
        "recommendation": rec,
        "best":         best,
    }

    os.makedirs(REPORT_DIR, exist_ok=True)

    with open(REPORT_JSON, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    buf = io.StringIO()
    with redirect_stdout(buf):
        print_comparison(comparison)
        print_per_entity(per_entity, verbose)
        print_annotation(annot_note)
        print_recommendation(rec)
    with open(REPORT_TXT, "w", encoding="utf-8") as fh:
        fh.write(buf.getvalue())

    print(f"\n  Reports saved:")
    print(f"    {REPORT_JSON}")
    print(f"    {REPORT_TXT}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NexResolve Entity Extraction Evaluation"
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    run(verbose=args.verbose)