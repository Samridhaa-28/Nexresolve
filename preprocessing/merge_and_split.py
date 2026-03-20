"""
NexResolve — Step 5: Merge + Split
Reads  : data/final/final_dataset.csv
Outputs: data/final/cleaned_issues.csv
         data/final/processed_comments.csv
         data/final/final_rl_dataset.csv
         data/splits/train.csv  val.csv  test.csv  knowledge_base.csv

Knowledge base strategy: ALL 852 issues included, tiered by quality
  tier1_verified  — closed + explicit solution comment  (weight 1.0)
  tier2_discussed — closed + >=2 comments, no solution tag (weight 0.7)
  tier3_minimal   — closed with little context          (weight 0.4)
"""

import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_SEED  = 42
TRAIN_RATIO  = 0.70
VAL_RATIO    = 0.15
MIN_COMMENTS = 2


def _dedup_cols(df):
    """Rename _x/_y suffixed columns from repeated merges, keep _x version."""
    rename = {}
    drop   = []
    for col in df.columns:
        if col.endswith("_x"):
            base = col[:-2]
            rename[col] = base
            y_col = base + "_y"
            if y_col in df.columns:
                drop.append(y_col)
    df = df.rename(columns=rename)
    df = df.drop(columns=[c for c in drop if c in df.columns])
    if "_frt" in df.columns:
        df = df.drop(columns=["_frt"])
    return df


def build_cleaned_issues(final):
    nlp_cols = [
        "repo", "issue_id", "issue_number",
        "clean_title", "clean_body", "clean_text",
        "all_comments_text", "solution_comments",
        "labels_normalised", "primary_label",
        "state", "created_at", "closed_at", "resolution_time_hours",
        "missing_version_flag", "missing_error_flag", "has_solution_comment",
        "text_length", "word_count",
        "urgency_keyword_flag", "question_mark_flag", "code_block_flag",
    ]
    available = [c for c in nlp_cols if c in final.columns]
    df = final[available].copy()
    # Fill text fields for issues with no comments (188 issues)
    # These are genuinely comment-free but should not be null — use empty string
    df["all_comments_text"] = df["all_comments_text"].fillna("")
    # solution_comments stays null intentionally for has_solution_comment=0 issues
    # so NLP pipeline can distinguish "no solution" from "empty solution"
    return df


def build_rl_dataset(final):
    rl_cols = [
        "issue_number", "primary_label",
        "text_length", "word_count",
        "missing_version_flag", "missing_error_flag",
        "urgency_keyword_flag", "question_mark_flag",
        "has_solution_comment",
        "turn_count", "reassignment_count", "reopen_count",
        "resolution_success", "sla_breach_flag", "sla_remaining_norm",
        "sla_limit_hours", "frustration_score", "interaction_depth",
        "first_response_time_hours",
    ]
    available = [c for c in rl_cols if c in final.columns]
    return final[available].copy()


def assign_kb_tier(row):
    """
    Tier each issue by resolution quality for weighted RAG retrieval.
    Retriever applies: tier1 x1.0 | tier2 x0.7 | tier3 x0.4
    knowledge_gap_flag fires only when NO tier1 result exceeds similarity threshold.

    Tier rules:
      tier1_verified  — has explicit solution comment  → best for suggestion action
      tier2_discussed — has real comment context        → good for similarity matching
      tier3_minimal   — no usable comment context       → clean_text only, low weight
    """
    if row.get("has_solution_comment", 0) == 1:
        return "tier1_verified"

    # Only tier2 if all_comments_text is actually populated
    # Issues where comments_count>=2 in raw but no rows in comments.csv
    # (bot-only or deleted) are downgraded to tier3
    has_real_comments = (
        isinstance(row.get("all_comments_text"), str)
        and str(row.get("all_comments_text", "")).strip() not in ("", "NO_COMMENTS", "nan")
        and len(str(row.get("all_comments_text", ""))) > 10
    )
    if has_real_comments:
        return "tier2_discussed"

    return "tier3_minimal"


def run(final_path           = "data/final/final_dataset.csv",
        comments_detail_path = "data/intermediate/cleaned_comments.csv",
        out_issues           = "data/final/cleaned_issues.csv",
        out_comments         = "data/final/processed_comments.csv",
        out_rl               = "data/final/final_rl_dataset.csv",
        split_dir            = "data/splits"):

    print("[merge_and_split] Loading final dataset ...")
    final = pd.read_csv(final_path)
    final = _dedup_cols(final)
    print(f"  Total rows: {len(final)}  cols: {len(final.columns)}")

    # ── NLP-ready issues ─────────────────────────────────────────────────
    ci = build_cleaned_issues(final)
    ci["all_comments_text"] = ci["all_comments_text"].fillna("NO_COMMENTS").astype(str)
    ci.to_csv(out_issues, index=False)
    print(f"  Saved cleaned_issues → {out_issues}  ({len(ci)} rows, {len(ci.columns)} cols)")

    # ── Processed comments with label context ────────────────────────────
    comments = pd.read_csv(comments_detail_path)
    if "primary_label" in final.columns:
        label_map = final.set_index("issue_number")["primary_label"].to_dict()
        comments["primary_label"] = comments["issue_number"].map(label_map)
    comments.to_csv(out_comments, index=False)
    print(f"  Saved processed_comments → {out_comments}  ({len(comments)} rows)")

    # ── RL dataset ────────────────────────────────────────────────────────
    rl = build_rl_dataset(final)
    rl.to_csv(out_rl, index=False)
    print(f"  Saved final_rl_dataset → {out_rl}  ({len(rl)} rows, {len(rl.columns)} cols)")

    # ── Train/Val/Test split ──────────────────────────────────────────────
    if "comments_count" in final.columns:
        trainable = final[final["comments_count"] >= MIN_COMMENTS].copy()
    else:
        trainable = final.copy()
    print(f"  Trainable issues (>=2 comments): {len(trainable)}")

    if "primary_label" in trainable.columns:
        counts    = trainable["primary_label"].value_counts()
        rare      = counts[counts < 3].index
        strat_col = trainable["primary_label"].where(
            ~trainable["primary_label"].isin(rare), other="other"
        )
    else:
        strat_col = None

    test_size = 1 - TRAIN_RATIO - VAL_RATIO
    train_val, test = train_test_split(
        trainable, test_size=test_size, random_state=RANDOM_SEED,
        stratify=strat_col
    )

    if strat_col is not None:
        tv_strat = train_val["primary_label"].where(
            ~train_val["primary_label"].isin(rare), other="other"
        )
    else:
        tv_strat = None

    val_size = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    train, val = train_test_split(
        train_val, test_size=val_size, random_state=RANDOM_SEED,
        stratify=tv_strat
    )

    # Fill all_comments_text nulls before writing splits
    # 188 issues had bot-only comments filtered out — use sentinel
    for df in [train, val, test]:
        df["all_comments_text"] = df["all_comments_text"].fillna("NO_COMMENTS")

    train.to_csv(f"{split_dir}/train.csv", index=False)
    val.to_csv(  f"{split_dir}/val.csv",   index=False)
    test.to_csv( f"{split_dir}/test.csv",  index=False)
    print(f"  Splits → train:{len(train)}  val:{len(val)}  test:{len(test)}")

    # ── Knowledge base for RAG (ALL 852 issues, tiered) ──────────────────
    #
    # Strategy: include every closed issue, tier by resolution quality.
    # Retriever uses kb_quality_tier for weighted similarity scoring.
    # knowledge_gap_flag = 1 only when NO tier1 result exceeds threshold.
    #
    kb = final.copy()

    # Remove junk issues — text too short to produce meaningful embeddings
    # Minimum 20 chars ensures at least a few meaningful words
    kb = kb[kb["clean_text"].str.len() >= 20].copy()
    junk_removed = len(final) - len(kb)
    if junk_removed > 0:
        print(f"  Removed {junk_removed} junk issues (clean_text < 20 chars) from KB")

    kb["kb_quality_tier"] = kb.apply(assign_kb_tier, axis=1)

    kb_keep = [c for c in [
        "issue_number", "clean_text", "all_comments_text",
        "solution_comments", "primary_label",
        "resolution_time_hours", "resolution_success",
        "has_solution_comment", "kb_quality_tier",
    ] if c in kb.columns]

    kb_out = kb[kb_keep].copy()
    # Fill nulls before writing — retriever must never receive null text fields
    kb_out["all_comments_text"] = kb_out["all_comments_text"].fillna("NO_COMMENTS")
    # solution_comments stays null for tier2/3 intentionally
    # retriever checks kb_quality_tier to know whether to use it
    kb_out.to_csv(f"{split_dir}/knowledge_base.csv", index=False)
    print(f"  Knowledge base → {split_dir}/knowledge_base.csv  ({len(kb)} issues)")
    print(f"  Tier distribution:\n{kb['kb_quality_tier'].value_counts().to_string()}")
    print("\n[merge_and_split] Done.")


if __name__ == "__main__":
    run()