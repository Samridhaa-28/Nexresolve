from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))



INTENT_REQUIRED_FIELDS: dict[str, set[str]] = {
    # Technical issues — need full reproduction environment
    "bug":         {"version", "error_type", "platform"},
    "ml_module":   {"version", "error_type", "hardware"},
    "build_infra": {"version", "platform"},

    # Documentation bug — which version has the wrong docs?
    "docs":        {"version"},

    # Non-technical — no reproduction needed
    "needs_info":  set(),   # already flagged as unclear
    "duplicate":   set(),   # just needs original issue ref
    "enhancement": set(),   # feature request, no repro
    "billing":     set(),   # business issue
    "other":       set(),   # catch-all
}

# Entity flag column names in the dataset
ENTITY_FLAG_COLS = {
    "version":    "has_version",
    "error_type": "has_error_type",
    "platform":   "has_platform",
    "hardware":   "has_hardware",
}

# Output columns written to CSVs
OUTPUT_COLS = [
    "missing_version",
    "missing_error",
    "missing_platform",
    "missing_hardware",
    "missing_count",
    "completeness_score",
]


# ─────────────────────────────────────────────────────────────────────────────
# CORE LOGIC — single issue
# ─────────────────────────────────────────────────────────────────────────────

def compute_missing(
    intent_group: str,
    has_version: int,
    has_error_type: int,
    has_platform: int,
    has_hardware: int,
) -> dict:
  
    required = INTENT_REQUIRED_FIELDS.get(intent_group, set())

    present_map = {
        "version":    int(has_version),
        "error_type": int(has_error_type),
        "platform":   int(has_platform),
        "hardware":   int(has_hardware),
    }

    # A field is "missing" only when:
    #   1. The intent REQUIRES it (not noise for irrelevant intents)
    #   2. The entity extractor did NOT find it (has_X = 0)
    missing_version  = int("version"    in required and present_map["version"]    == 0)
    missing_error    = int("error_type" in required and present_map["error_type"] == 0)
    missing_platform = int("platform"   in required and present_map["platform"]   == 0)
    missing_hardware = int("hardware"   in required and present_map["hardware"]   == 0)

    missing_count = missing_version + missing_error + missing_platform + missing_hardware

    # completeness_score: fraction of required fields that ARE present
    # If nothing is required (billing/duplicate/etc.) → 1.0 by default
    # This is the correct behaviour — a billing ticket with no version is COMPLETE
    if len(required) == 0:
        completeness_score = 1.0
    else:
        present_required = sum(
            1 for field in required
            if present_map.get(field, 0) == 1
        )
        completeness_score = round(present_required / len(required), 4)

    return {
        "missing_version":    missing_version,
        "missing_error":      missing_error,
        "missing_platform":   missing_platform,
        "missing_hardware":   missing_hardware,
        "missing_count":      missing_count,
        "completeness_score": completeness_score,
    }


def compute_missing_batch(df: pd.DataFrame) -> pd.DataFrame:
   
    _check_prerequisites(df)

    rows = []
    for _, row in df.iterrows():
        result = compute_missing(
            intent_group  = str(row.get("intent_group", "other")),
            has_version   = int(row.get("has_version",    0)),
            has_error_type= int(row.get("has_error_type", 0)),
            has_platform  = int(row.get("has_platform",   0)),
            has_hardware  = int(row.get("has_hardware",   0)),
        )
        rows.append(result)

    return pd.DataFrame(rows)


def _check_prerequisites(df: pd.DataFrame) -> None:
  
    required_cols = ["intent_group", "has_version", "has_error_type",
                     "has_platform", "has_hardware"]
    missing_cols  = [c for c in required_cols if c not in df.columns]

    if missing_cols:
        msg = (
            f"\nMissing columns: {missing_cols}\n\n"
            "Run these first:\n"
        )
        if "intent_group" in missing_cols:
            msg += "  python nlp/confidence_estimator.py --mode generate --model best\n"
        if any(c.startswith("has_") for c in missing_cols):
            msg += "  python nlp/entity_extractor.py --mode generate\n"
        raise ValueError(msg)


# ─────────────────────────────────────────────────────────────────────────────
# GENERATE FEATURES — write to CSVs
# ─────────────────────────────────────────────────────────────────────────────

def generate_features(
    issues_path: str = "data/final/cleaned_issues.csv",
    rl_path: str     = "data/final/final_rl_dataset.csv",
) -> pd.DataFrame:
    
    print("Loading data ...")
    ci = pd.read_csv(issues_path, encoding="utf-8")
    rl = pd.read_csv(rl_path,     encoding="utf-8")
    print(f"  Issues: {len(ci)}")

    # ── Build working DataFrame ──────────────────────────────────────────────
    # intent_group may be in CI but not RL — merge carefully
    work = ci[["issue_number"]].copy()

    # intent_group
    if "intent_group" in ci.columns:
        work["intent_group"] = ci["intent_group"].values
    elif "intent_group" in rl.columns:
        work = work.merge(rl[["issue_number","intent_group"]],
                          on="issue_number", how="left")
    else:
        raise ValueError(
            "intent_group not found in either CSV.\n"
            "Run: python nlp/confidence_estimator.py --mode generate --model best"
        )

    # entity flags — prefer CI (more complete), fall back to RL
    for col in ["has_version","has_error_type","has_platform","has_hardware"]:
        if col in ci.columns:
            work[col] = ci[col].values
        elif col in rl.columns:
            work = work.merge(rl[["issue_number", col]],
                              on="issue_number", how="left")
        else:
            raise ValueError(
                f"{col} not found.\n"
                "Run: python nlp/entity_extractor.py --mode generate"
            )

    # ── Compute missing flags ────────────────────────────────────────────────
    print("Computing missing flags ...")
    missing_df = compute_missing_batch(work)
    missing_df["issue_number"] = work["issue_number"].values

    old_flags       = ["missing_version_flag", "missing_error_flag"]
    old_flags_rl_only = ["urgency_keyword_flag"]

    # ── Update cleaned_issues.csv ────────────────────────────────────────────
    for col in old_flags + OUTPUT_COLS:
        if col in ci.columns:
            ci = ci.drop(columns=[col])
    for col in old_flags:
        if col in ci.columns:
            print(f"  Removed old column from CI: {col}")
    ci = ci.merge(missing_df[["issue_number"] + OUTPUT_COLS],
                  on="issue_number", how="left")
    ci.to_csv(issues_path, index=False, encoding="utf-8")
    print(f"  Updated → {issues_path}")

    # ── Update final_rl_dataset.csv ──────────────────────────────────────────
    for col in old_flags + old_flags_rl_only:
        if col in rl.columns:
            rl = rl.drop(columns=[col])
            print(f"  Removed placeholder column from RL: {col}")

    # Remove any existing output cols before merging
    for col in OUTPUT_COLS:
        if col in rl.columns:
            rl = rl.drop(columns=[col])

    rl = rl.merge(missing_df[["issue_number"] + OUTPUT_COLS],
                  on="issue_number", how="left")
    rl.to_csv(rl_path, index=False, encoding="utf-8")
    print(f"  Updated → {rl_path}")

    # ── Summary ──────────────────────────────────────────────────────────────
    _print_summary(missing_df, work)

    return missing_df[["issue_number"] + OUTPUT_COLS]


def _print_summary(missing_df: pd.DataFrame, work: pd.DataFrame) -> None:
    """Print distribution summary after generation."""
    print()
    print("=" * 57)
    print("  MISSING DETECTION SUMMARY")
    print("=" * 57)
    print()

    total = len(missing_df)

    # Overall flag rates
    print(f"  {'Flag':<22} {'Count':>6} {'%':>7}")
    print("  " + "─" * 37)
    for col in ["missing_version","missing_error","missing_platform","missing_hardware"]:
        n   = int(missing_df[col].sum())
        pct = n / total * 100
        print(f"  {col:<22} {n:>6} {pct:>6.1f}%")

    print()
    print(f"  {'missing_count':<22} mean={missing_df['missing_count'].mean():.2f}")
    print(f"  {'completeness_score':<22} mean={missing_df['completeness_score'].mean():.3f}")

    # Per-intent completeness
    print()
    print("  Completeness by intent group:")
    print(f"  {'Intent':<15} {'n':>5} {'complete':>9} {'avg_score':>10}")
    print("  " + "─" * 42)

    work_full = work.copy()
    work_full["completeness_score"] = missing_df["completeness_score"].values
    work_full["missing_count"]      = missing_df["missing_count"].values

    for intent in sorted(INTENT_REQUIRED_FIELDS.keys()):
        subset = work_full[work_full["intent_group"] == intent]
        if len(subset) == 0:
            continue
        fully_complete = (subset["missing_count"] == 0).sum()
        avg_score      = subset["completeness_score"].mean()
        required       = INTENT_REQUIRED_FIELDS[intent]
        req_str        = ", ".join(sorted(required)) if required else "—"
        print(
            f"  {intent:<15} {len(subset):>5} "
            f"{fully_complete:>6} ({fully_complete/len(subset)*100:.0f}%)"
            f" {avg_score:>10.3f}  requires: {req_str}"
        )

    # Missing count distribution
    print()
    print("  Missing count distribution:")
    for n in range(5):
        count = (missing_df["missing_count"] == n).sum()
        bar   = "█" * int(count / total * 30)
        print(f"    {n} missing: {count:4d} ({count/total*100:.1f}%)  |{bar}")

    print()
    print("  RL Agent interpretation:")
    print("    completeness=1.0 → all info present → safe to Route or Suggest")
    print("    completeness=0.5 → partial info → Route with caution")
    print("    completeness=0.0 → nothing present → must Clarify first")


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE — single issue
# ─────────────────────────────────────────────────────────────────────────────

def predict(
    intent_group: str,
    has_version: int    = 0,
    has_error_type: int = 0,
    has_platform: int   = 0,
    has_hardware: int   = 0,
) -> dict:
    
    result = compute_missing(
        intent_group, has_version, has_error_type, has_platform, has_hardware
    )

    required = INTENT_REQUIRED_FIELDS.get(intent_group, set())

    # Build human-readable summary
    missing_fields = []
    if result["missing_version"]:
        missing_fields.append("version")
    if result["missing_error"]:
        missing_fields.append("error_type")
    if result["missing_platform"]:
        missing_fields.append("platform")
    if result["missing_hardware"]:
        missing_fields.append("hardware")

    if not required:
        summary = f"Intent '{intent_group}' requires no specific fields — complete by default"
    elif not missing_fields:
        summary = f"All required fields present ({', '.join(sorted(required))})"
    else:
        summary = f"Missing: {', '.join(missing_fields)} (required for {intent_group})"

    result["required_fields"] = sorted(required)
    result["missing_fields"]  = missing_fields
    result["summary"]         = summary
    return result


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS — for notebook
# ─────────────────────────────────────────────────────────────────────────────

def analyse(
    issues_path: str = "data/final/cleaned_issues.csv",
    rl_path: str     = "data/final/final_rl_dataset.csv",
) -> dict:
    """
    Run distribution analysis on generated missing flags.
    Returns a dict of results for the notebook to visualise.
    """
    ci = pd.read_csv(issues_path, encoding="utf-8")
    rl = pd.read_csv(rl_path,     encoding="utf-8")

    missing_cols = ["missing_version","missing_error","missing_platform",
                    "missing_hardware","missing_count","completeness_score"]
    if not all(c in ci.columns for c in missing_cols):
        raise ValueError(
            "Missing flags not found in cleaned_issues.csv.\n"
            "Run: python nlp/missing_detector.py first"
        )

    # Per-intent completeness
    intent_stats = {}
    if "intent_group" in ci.columns:
        for intent in INTENT_REQUIRED_FIELDS:
            subset = ci[ci["intent_group"] == intent]
            if len(subset) == 0:
                continue
            intent_stats[intent] = {
                "n":                  int(len(subset)),
                "avg_completeness":   round(subset["completeness_score"].mean(), 3),
                "pct_complete":       round((subset["missing_count"] == 0).mean() * 100, 1),
                "avg_missing_count":  round(subset["missing_count"].mean(), 2),
                "required_fields":    sorted(INTENT_REQUIRED_FIELDS[intent]),
            }

    # Correlation: completeness vs resolution success
    corr_data = {}
    if "resolution_success" in rl.columns:
        merged = ci.merge(rl[["issue_number","resolution_success"]],
                          on="issue_number", how="left")
        corr = merged["completeness_score"].corr(merged["resolution_success"])
        corr_data["completeness_vs_success_corr"] = round(corr, 4)

        # By completeness bucket
        merged["completeness_bucket"] = pd.cut(
            merged["completeness_score"],
            bins=[0, 0.25, 0.5, 0.75, 1.01],
            labels=["0-25%","25-50%","50-75%","75-100%"],
            include_lowest=True,
        )
        bucket_stats = merged.groupby("completeness_bucket", observed=True).agg(
            avg_success=("resolution_success","mean"),
            n=("resolution_success","count"),
        ).round(3).to_dict()
        corr_data["bucket_stats"] = bucket_stats

    return {
        "flag_rates":        {c: round(ci[c].mean(), 4) for c in missing_cols[:4]},
        "missing_count_dist":{int(k): int(v)
                              for k, v in ci["missing_count"].value_counts().sort_index().items()},
        "completeness_dist": {
            "mean":   round(ci["completeness_score"].mean(), 3),
            "median": round(ci["completeness_score"].median(), 3),
            "pct_1":  round((ci["completeness_score"] == 1.0).mean() * 100, 1),
            "pct_0":  round((ci["completeness_score"] == 0.0).mean() * 100, 1),
        },
        "intent_stats":  intent_stats,
        "correlation":   corr_data,
        "required_fields": {k: sorted(v) for k, v in INTENT_REQUIRED_FIELDS.items()},
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="NexResolve Missing Information Detector",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--issues_path", default="data/final/cleaned_issues.csv")
    parser.add_argument("--rl_path",     default="data/final/final_rl_dataset.csv")
    parser.add_argument("--mode",        default="generate",
                        choices=["generate", "analyse", "predict"])
    parser.add_argument("--intent",      default="bug",
                        help="Intent group for --mode predict")
    parser.add_argument("--has_version",    type=int, default=0)
    parser.add_argument("--has_error",      type=int, default=0)
    parser.add_argument("--has_platform",   type=int, default=0)
    parser.add_argument("--has_hardware",   type=int, default=0)
    args = parser.parse_args()

    if args.mode == "predict":
        import json
        result = predict(
            intent_group   = args.intent,
            has_version    = args.has_version,
            has_error_type = args.has_error,
            has_platform   = args.has_platform,
            has_hardware   = args.has_hardware,
        )
        print(json.dumps(result, indent=2))

    elif args.mode == "analyse":
        import json
        result = analyse(args.issues_path, args.rl_path)
        print(json.dumps(result, indent=2, default=str))

    elif args.mode == "generate":
        generate_features(args.issues_path, args.rl_path)
        print("\n✓ Missing detection complete.")
        print("  Next steps:")
        print("    python nlp/missing_detector.py --mode analyse")
        print("    jupyter notebook notebooks/06_missing_detection.ipynb")
        print("    python nlp/clarification_modeler.py")


if __name__ == "__main__":
    main()