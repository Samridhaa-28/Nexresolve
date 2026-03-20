

import sys
import os
import shutil
import time

# ── Path setup ───────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ── Import pipeline steps ────────────────────────────────────────────────────
from preprocessing.clean_issues       import run as clean_issues
from preprocessing.clean_comments     import run as clean_comments
from preprocessing.clean_events       import run as clean_events
from preprocessing.feature_engineering import run as feature_engineering
from preprocessing.merge_and_split    import run as merge_and_split


def ensure_dirs():
    for d in [
        "data/raw", "data/intermediate", "data/final", "data/splits",
        "models/nlp", "models/rl", "models/embeddings",
    ]:
        os.makedirs(d, exist_ok=True)


def copy_raw_files(src_issues, src_comments, src_events):
    """Copy uploaded CSVs into data/raw/ if they're not already there."""
    for src, dst in [
        (src_issues,   "data/raw/issues.csv"),
        (src_comments, "data/raw/comments.csv"),
        (src_events,   "data/raw/events.csv"),
    ]:
        if src and os.path.exists(src) and not os.path.exists(dst):
            shutil.copy(src, dst)
            print(f"  Copied {src} → {dst}")


def run(
    issues_src   = None,
    comments_src = None,
    events_src   = None,
):
    t0 = time.time()
    print("=" * 60)
    print("NexResolve Preprocessing Pipeline")
    print("=" * 60)

    ensure_dirs()
    copy_raw_files(issues_src, comments_src, events_src)

    # ── STEP 1 ───────────────────────────────────────────────────────────
    print("\n[STEP 1] Cleaning issues ...")
    clean_issues(
        input_path  = "data/raw/issues.csv",
        output_path = "data/intermediate/cleaned_issues.csv",
    )

    # ── STEP 2 ───────────────────────────────────────────────────────────
    print("\n[STEP 2] Cleaning comments ...")
    clean_comments(
        input_path  = "data/raw/comments.csv",
        issues_path = "data/intermediate/cleaned_issues.csv",
        output_path = "data/intermediate/cleaned_comments.csv",
        agg_path    = "data/intermediate/aggregated_comments.csv",
    )

    # ── STEP 3 ───────────────────────────────────────────────────────────
    print("\n[STEP 3] Cleaning events ...")
    clean_events(
        input_path  = "data/raw/events.csv",
        output_path = "data/intermediate/cleaned_events.csv",
        agg_path    = "data/intermediate/aggregated_events.csv",
    )

    # ── STEP 4 ───────────────────────────────────────────────────────────
    print("\n[STEP 4] Feature engineering ...")
    feature_engineering(
        issues_path          = "data/intermediate/cleaned_issues.csv",
        comments_path        = "data/intermediate/aggregated_comments.csv",
        events_path          = "data/intermediate/aggregated_events.csv",
        comments_detail_path = "data/intermediate/cleaned_comments.csv",
        nlp_out              = "data/intermediate/features_nlp.csv",
        rl_out               = "data/intermediate/features_rl.csv",
        final_out            = "data/final/final_dataset.csv",
    )

    # ── STEP 5 ───────────────────────────────────────────────────────────
    print("\n[STEP 5] Merge and split ...")
    merge_and_split(
        final_path           = "data/final/final_dataset.csv",
        comments_detail_path = "data/intermediate/cleaned_comments.csv",
        out_issues           = "data/final/cleaned_issues.csv",
        out_comments         = "data/final/processed_comments.csv",
        out_rl               = "data/final/final_rl_dataset.csv",
        split_dir            = "data/splits",
    )

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"Pipeline complete in {elapsed:.1f}s")
    print("Output files:")
    for f in [
        "data/final/cleaned_issues.csv",
        "data/final/processed_comments.csv",
        "data/final/final_rl_dataset.csv",
        "data/splits/train.csv",
        "data/splits/val.csv",
        "data/splits/test.csv",
        "data/splits/knowledge_base.csv",
    ]:
        if os.path.exists(f):
            size = os.path.getsize(f)
            rows = sum(1 for _ in open(f)) - 1
            print(f"  ✓ {f}  ({rows} rows, {size//1024}KB)")
        else:
            print(f"  ✗ MISSING: {f}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--issues",   default=None)
    p.add_argument("--comments", default=None)
    p.add_argument("--events",   default=None)
    args = p.parse_args()
    run(issues_src=args.issues, comments_src=args.comments, events_src=args.events)