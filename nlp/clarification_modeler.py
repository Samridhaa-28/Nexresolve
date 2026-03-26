"""
NexResolve — Clarification Modeler (Dual-Trigger Policy)
nlp/clarification_modeler.py

Determines whether a ticket needs clarification using two triggers:
  TRIGGER 1 — Uncertainty:    uncertainty_flag=1, intent not reliable
  TRIGGER 2 — Missing entity: intent confident, required field absent

DECISION FLOW
-------------
  1. needs_info   → ALWAYS clarify (label means ticket is too vague)
  2. duplicate    → NEVER clarify  (use RAG/retrieval instead)
  3. other        → NEVER clarify  (noise/invalid, escalate)
  4. uncertainty_flag=1 → generic semantic clarification question
  5. billing (confident)  → no clarification, route_billing directly
  6. enhancement (vague)  → ask to elaborate on feature request
  7. enhancement (clear)  → no clarification, route_product
  8. technical + missing  → entity-specific question
  9. technical + complete → no clarification, route_technical

PRIORITY ENCODING (clarification_priority)
------------------------------------------
  0 = none          (no clarification needed)
  1 = uncertainty   (generic semantic — highest priority)
  2 = needs_info    (always clarify)
  3 = error_type    (most diagnostic entity)
  4 = version
  5 = platform
  6 = hardware
  7 = vague_request (enhancement too short)

REFINEMENTS ADDED
-----------------
  1. clarification_trigger stored in cleaned_issues.csv (not RL dataset)
     Gives human-readable WHY: uncertainty/missing_entity/needs_info/
     vague_request/none. RL state unchanged.

  2. Enhancement vagueness: word_count < 15 threshold.

  3. Billing: confident billing always routes directly — no clarification.
     Our entity extractor covers tech entities only.

  4. Priority 0-7 (extended from 0-4) — backward compatible ordinal feature.

OUTPUT
------
  cleaned_issues.csv:
    clarification_question, clarification_type, clarification_trigger,
    needs_clarification, clarification_priority

  final_rl_dataset.csv:
    needs_clarification, clarification_priority  (numeric only)

USAGE
-----
  python nlp/clarification_modeler.py
  python nlp/clarification_modeler.py --mode predict --intent bug
      --uncertainty_flag 1
  python nlp/clarification_modeler.py --mode predict --intent bug
      --uncertainty_flag 0 --missing_version 1 --missing_error 1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

CLARIFICATION_PRIORITY: dict[str, int] = {
    "none":          0,
    "uncertainty":   1,
    "needs_info":    2,
    "error_type":    3,
    "version":       4,
    "platform":      5,
    "hardware":      6,
    "vague_request": 7,
}

PRIORITY_TO_TYPE: dict[int, str] = {v: k for k, v in CLARIFICATION_PRIORITY.items()}

# Enhancement requests shorter than this are considered too vague
ENHANCEMENT_VAGUE_THRESHOLD: int = 15

# Intents that require no clarification regardless of missing fields
NEVER_CLARIFY = {"duplicate", "other"}

# Intents that always clarify
ALWAYS_CLARIFY = {"needs_info"}

# Technical intents that use entity-based clarification
TECHNICAL_INTENTS = {"bug", "ml_module", "build_infra", "docs"}


# ─────────────────────────────────────────────────────────────────────────────
# QUESTION TEMPLATES
# ─────────────────────────────────────────────────────────────────────────────

UNCERTAINTY_QUESTIONS: dict[str, str] = {
    "bug": (
        "We are not entirely sure what type of issue this is. "
        "Could you describe the problem in more detail — what you were trying "
        "to do and what went wrong?"
    ),
    "ml_module": (
        "Could you provide more context about the issue with the ML framework? "
        "Please describe the behaviour you expected versus what actually happened."
    ),
    "build_infra": (
        "Could you describe the build or installation issue in more detail? "
        "What command did you run and what error did you see?"
    ),
    "billing": (
        "Could you confirm whether this is a billing or payment-related issue? "
        "A brief description will help us route it correctly."
    ),
    "enhancement": (
        "Could you describe the feature or improvement you are requesting? "
        "A clearer description helps us understand what you need."
    ),
    "docs": (
        "Could you point to the specific documentation page or section "
        "that contains the error or inconsistency?"
    ),
    "_default": (
        "Could you describe your issue in more detail? "
        "This will help us understand what you need and route it correctly."
    ),
}

ENTITY_QUESTIONS: dict[str, dict[str, str]] = {
    "bug": {
        "error_type": (
            "Could you share the exact error message or stack trace? "
            "This is the most important detail for diagnosing the issue."
        ),
        "version": (
            "What version of the software are you using? "
            "This helps us check if the issue has already been fixed."
        ),
        "platform": (
            "What operating system are you running on? "
            "(e.g. Windows 11, Ubuntu 22.04, macOS 14)"
        ),
        "hardware": (
            "What GPU or hardware device are you using? "
            "(e.g. NVIDIA RTX 3090, AMD RX 6900, Apple M2)"
        ),
    },
    "ml_module": {
        "error_type": (
            "What is the exact error message you are seeing? "
            "For CUDA or framework errors, please include the full traceback."
        ),
        "version": (
            "What versions of PyTorch/TensorFlow and CUDA are you using? "
            "(e.g. PyTorch 2.1.0, CUDA 11.8, Python 3.10)"
        ),
        "platform": (
            "What operating system are you running on? "
            "(e.g. Ubuntu 22.04, Windows 11, macOS 14)"
        ),
        "hardware": (
            "What GPU or hardware device are you using? "
            "(e.g. NVIDIA A100, RTX 3090, AMD ROCm, Apple MPS)"
        ),
    },
    "build_infra": {
        "error_type": (
            "Could you share the exact error from the build or install process? "
            "Please include any compiler or linker errors."
        ),
        "version": (
            "What version of the package are you trying to install, "
            "and what Python/pip version are you using?"
        ),
        "platform": (
            "What operating system and architecture are you building on? "
            "(e.g. Ubuntu 22.04 x86_64, Windows 11 ARM64)"
        ),
        "hardware": (
            "What GPU or compute hardware are you targeting? "
            "(e.g. NVIDIA CUDA, AMD ROCm, CPU-only)"
        ),
    },
    "docs": {
        "version": (
            "Which version of the documentation or software are you referring to? "
            "The docs may differ between versions."
        ),
        "error_type": (
            "Could you describe the specific error or inconsistency in the docs?"
        ),
        "platform": (
            "What operating system are you on? "
            "Some documentation sections are platform-specific."
        ),
        "hardware": (
            "What hardware are you using? This may affect which docs apply."
        ),
    },
    "_default": {
        "error_type": "Could you share the exact error message you are encountering?",
        "version":    "What version of the software are you using?",
        "platform":   "What operating system are you running on?",
        "hardware":   "What GPU or hardware device are you using?",
    },
}

NEEDS_INFO_QUESTION = (
    "Could you describe your issue in more detail? "
    "Please include what you were trying to do, what happened, "
    "and any error messages you saw."
)

VAGUE_ENHANCEMENT_QUESTION = (
    "Could you describe the feature you are requesting in more detail? "
    "What problem would it solve and how do you envision it working?"
)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _get_uncertainty_q(intent: str) -> str:
    return UNCERTAINTY_QUESTIONS.get(intent, UNCERTAINTY_QUESTIONS["_default"])


def _get_entity_q(intent: str, field: str) -> str:
    t = ENTITY_QUESTIONS.get(intent, ENTITY_QUESTIONS["_default"])
    return t.get(field, ENTITY_QUESTIONS["_default"].get(field, ""))


def _result(question: str, ctype: str, trigger: str, priority: int) -> dict:
    return {
        "clarification_question": question,
        "clarification_type":     ctype,
        "clarification_trigger":  trigger,
        "needs_clarification":    int(priority > 0),
        "clarification_priority": priority,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CORE LOGIC — single issue
# ─────────────────────────────────────────────────────────────────────────────

def generate_clarification(
    intent_group:     str,
    uncertainty_flag: int,
    missing_version:  int = 0,
    missing_error:    int = 0,
    missing_platform: int = 0,
    missing_hardware: int = 0,
    word_count:       int = 50,
) -> dict:
    """
    Dual-trigger clarification policy for a single issue.
    See module docstring for complete decision flow.
    """
    # ── Step 1: needs_info — always clarify ──────────────────────────────────
    if intent_group in ALWAYS_CLARIFY:
        return _result(NEEDS_INFO_QUESTION, "needs_info", "needs_info",
                       CLARIFICATION_PRIORITY["needs_info"])

    # ── Step 2-3: duplicate / other — never clarify ──────────────────────────
    if intent_group in NEVER_CLARIFY:
        return _result("", "none", "none", CLARIFICATION_PRIORITY["none"])

    # ── Step 4: uncertainty — generic question ───────────────────────────────
    # Must come BEFORE entity checks.
    # When intent is uncertain, asking "what version?" may be wrong intent.
    if uncertainty_flag == 1:
        return _result(_get_uncertainty_q(intent_group),
                       "uncertainty", "uncertainty",
                       CLARIFICATION_PRIORITY["uncertainty"])

    # ── From here: intent is confident (uncertainty_flag = 0) ────────────────

    # ── Step 5: billing (confident) — route directly ─────────────────────────
    if intent_group == "billing":
        return _result("", "none", "none", CLARIFICATION_PRIORITY["none"])

    # ── Step 6-7: enhancement ────────────────────────────────────────────────
    if intent_group == "enhancement":
        if word_count < ENHANCEMENT_VAGUE_THRESHOLD:
            return _result(VAGUE_ENHANCEMENT_QUESTION, "vague_request",
                           "vague_request", CLARIFICATION_PRIORITY["vague_request"])
        return _result("", "none", "none", CLARIFICATION_PRIORITY["none"])

    # ── Step 8-9: technical intents ──────────────────────────────────────────
    if intent_group in TECHNICAL_INTENTS:
        missing = {}
        if missing_error:    missing["error_type"] = CLARIFICATION_PRIORITY["error_type"]
        if missing_version:  missing["version"]    = CLARIFICATION_PRIORITY["version"]
        if missing_platform: missing["platform"]   = CLARIFICATION_PRIORITY["platform"]
        if missing_hardware: missing["hardware"]   = CLARIFICATION_PRIORITY["hardware"]

        if missing:
            top = min(missing, key=lambda f: missing[f])
            return _result(_get_entity_q(intent_group, top),
                           top, "missing_entity", missing[top])

        return _result("", "none", "none", CLARIFICATION_PRIORITY["none"])

    # ── Fallback ──────────────────────────────────────────────────────────────
    return _result("", "none", "none", CLARIFICATION_PRIORITY["none"])


def generate_clarification_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Run generate_clarification on an entire DataFrame."""
    _check_prerequisites(df)
    rows = []
    for _, row in df.iterrows():
        rows.append(generate_clarification(
            intent_group     = str(row.get("intent_group",     "other")),
            uncertainty_flag = int(row.get("uncertainty_flag", 0)),
            missing_version  = int(row.get("missing_version",  0)),
            missing_error    = int(row.get("missing_error",    0)),
            missing_platform = int(row.get("missing_platform", 0)),
            missing_hardware = int(row.get("missing_hardware", 0)),
            word_count       = int(row.get("word_count",       50)),
        ))
    return pd.DataFrame(rows)


def _check_prerequisites(df: pd.DataFrame) -> None:
    required = ["intent_group", "uncertainty_flag",
                "missing_version", "missing_error",
                "missing_platform", "missing_hardware"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        msg = f"\nMissing columns: {missing}\n\nRun these first:\n"
        if "intent_group" in missing or "uncertainty_flag" in missing:
            msg += "  python nlp/confidence_estimator.py --mode generate --model best\n"
        if any("missing_" in c for c in missing):
            msg += "  python nlp/missing_detector.py\n"
        raise ValueError(msg)


# ─────────────────────────────────────────────────────────────────────────────
# GENERATE FEATURES
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_COLS_CI = [
    "clarification_question",
    "clarification_type",
    "clarification_trigger",
    "needs_clarification",
    "clarification_priority",
]

OUTPUT_COLS_RL = [
    "needs_clarification",
    "clarification_priority",
]


def generate_features(
    issues_path: str = "data/final/cleaned_issues.csv",
    rl_path:     str = "data/final/final_rl_dataset.csv",
) -> pd.DataFrame:
    """Write clarification flags to both CSVs."""
    print("Loading data ...")
    ci = pd.read_csv(issues_path, encoding="utf-8")
    rl = pd.read_csv(rl_path,     encoding="utf-8")
    print(f"  Issues: {len(ci)}")

    needed = ["issue_number", "intent_group", "uncertainty_flag",
              "missing_version", "missing_error",
              "missing_platform", "missing_hardware"]
    for col in needed:
        if col not in ci.columns:
            raise ValueError(
                f"Column '{col}' not found.\n"
                "Run prerequisites:\n"
                "  python nlp/confidence_estimator.py --mode generate --model best\n"
                "  python nlp/entity_extractor.py --mode generate\n"
                "  python nlp/missing_detector.py"
            )

    work_cols = needed + (["word_count"] if "word_count" in ci.columns else [])
    work = ci[work_cols].copy()

    print("Generating clarification decisions ...")
    clar_df = generate_clarification_batch(work)
    clar_df["issue_number"] = work["issue_number"].values

    # Update cleaned_issues.csv
    for col in OUTPUT_COLS_CI:
        if col in ci.columns:
            ci = ci.drop(columns=[col])
    ci = ci.merge(clar_df[["issue_number"] + OUTPUT_COLS_CI],
                  on="issue_number", how="left")
    ci.to_csv(issues_path, index=False, encoding="utf-8")
    print(f"  Updated → {issues_path}")

    # Update final_rl_dataset.csv
    for col in OUTPUT_COLS_RL:
        if col in rl.columns:
            rl = rl.drop(columns=[col])
    rl = rl.merge(clar_df[["issue_number"] + OUTPUT_COLS_RL],
                  on="issue_number", how="left")
    rl.to_csv(rl_path, index=False, encoding="utf-8")
    print(f"  Updated → {rl_path}")

    _print_summary(clar_df, work)
    return clar_df[["issue_number"] + OUTPUT_COLS_CI]


def _print_summary(clar_df: pd.DataFrame, work: pd.DataFrame) -> None:
    total = len(clar_df)
    needs = clar_df["needs_clarification"].sum()

    print()
    print("=" * 57)
    print("  CLARIFICATION MODELER — DUAL TRIGGER POLICY")
    print("=" * 57)
    print(f"\n  Total:               {total}")
    print(f"  Needs clarification: {needs} ({needs/total*100:.1f}%)")
    print(f"  No clarification:    {total-needs} ({(total-needs)/total*100:.1f}%)")
    print()
    print("  By trigger:")
    for t in ["needs_info","uncertainty","missing_entity","vague_request","none"]:
        n = (clar_df["clarification_trigger"] == t).sum()
        b = "█" * int(n/total*25)
        print(f"    {t:15s}: {n:4d} ({n/total*100:.1f}%)  |{b}")

    print()
    print("  By intent:")
    work_full = work.copy()
    work_full["nc"] = clar_df["needs_clarification"].values
    work_full["trig"] = clar_df["clarification_trigger"].values
    for intent in sorted(work_full["intent_group"].unique()):
        sub = work_full[work_full["intent_group"] == intent]
        rate = sub["nc"].mean() * 100
        n    = len(sub)
        trigs = sub[sub["nc"]==1]["trig"]
        top  = trigs.mode()[0] if len(trigs) > 0 else "—"
        b = "█" * int(rate/100*15)
        print(f"    {intent:15s} (n={n:3d}): {rate:5.1f}%  |{b}  [{top}]")


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

def predict(
    intent_group:     str,
    uncertainty_flag: int = 0,
    missing_version:  int = 0,
    missing_error:    int = 0,
    missing_platform: int = 0,
    missing_hardware: int = 0,
    word_count:       int = 50,
) -> dict:
    """Single-ticket inference. Called by nlp_pipeline.py."""
    result = generate_clarification(
        intent_group, uncertainty_flag,
        missing_version, missing_error,
        missing_platform, missing_hardware,
        word_count,
    )
    if result["needs_clarification"] == 0:
        result["summary"] = "No clarification needed"
    else:
        result["summary"] = (
            f"Trigger: {result['clarification_trigger']} → "
            f"ask about {result['clarification_type']}"
        )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="NexResolve Clarification Modeler (Dual-Trigger)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode",             default="generate",
                        choices=["generate","predict"])
    parser.add_argument("--issues_path",      default="data/final/cleaned_issues.csv")
    parser.add_argument("--rl_path",          default="data/final/final_rl_dataset.csv")
    parser.add_argument("--intent",           default="bug")
    parser.add_argument("--uncertainty_flag", type=int, default=0)
    parser.add_argument("--missing_version",  type=int, default=0)
    parser.add_argument("--missing_error",    type=int, default=0)
    parser.add_argument("--missing_platform", type=int, default=0)
    parser.add_argument("--missing_hardware", type=int, default=0)
    parser.add_argument("--word_count",       type=int, default=50)
    args = parser.parse_args()

    if args.mode == "predict":
        result = predict(
            args.intent, args.uncertainty_flag,
            args.missing_version, args.missing_error,
            args.missing_platform, args.missing_hardware,
            args.word_count,
        )
        print(json.dumps(result, indent=2))

    else:
        generate_features(args.issues_path, args.rl_path)
        print("\n✓ Done.")
        print("  jupyter notebook notebooks/06_missing_detection.ipynb")


if __name__ == "__main__":
    main()