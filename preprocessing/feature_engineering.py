

import re
import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════
# SLA THRESHOLDS (hours) — adjust to your domain
# ═══════════════════════════════════════════════════════════
SLA_LIMITS = {
    "bug":         48.0,
    "question":    72.0,
    "triage":      24.0,
    "billing":     24.0,
    "enhancement": 168.0,
    "default":     72.0,
}


def get_sla_limit(primary_label: str) -> float:
    return SLA_LIMITS.get(primary_label, SLA_LIMITS["default"])


# ═══════════════════════════════════════════════════════════
# RL FEATURES
# ═══════════════════════════════════════════════════════════

def compute_rl_features(issues: pd.DataFrame,
                        agg_comments: pd.DataFrame,
                        agg_events: pd.DataFrame) -> pd.DataFrame:
    """
    Computes all RL-required derived features.

    Features:
        turn_count           — number of comment turns per issue
        reassignment_count   — number of assignment events
        reopen_count         — number of reopen events
        resolution_success   — 1 if resolved (closed + has_solution_comment)
        sla_breach_flag      — 1 if resolution_time > SLA threshold for label
        sla_remaining_norm   — (SLA_limit - resolution_time) / SLA_limit, clipped [0,1]
        frustration_score    — proxy: turns * (1 + reassignment) * reopen penalty
        interaction_depth    — unique commenters + reassignment_count
    """
    df = issues.copy()
    # Drop placeholder to avoid collision with aggregated version
    if "has_solution_comment" in df.columns:
        df = df.drop(columns=["has_solution_comment"])

    # Merge comment aggregates
    df = df.merge(
        agg_comments[["issue_number", "comment_count", "has_solution_comment",
                       "unique_commenters"]],
        on="issue_number", how="left", suffixes=("", "_c")
    )

    # Merge event aggregates
    df = df.merge(
        agg_events[["issue_number", "reassignment_count", "reopen_count",
                    "unique_actors"]],
        on="issue_number", how="left"
    )

    # Fill NaN from unmatched merges
    df["comment_count"]       = df["comment_count"].fillna(0).astype(int)
    df["has_solution_comment"]= df["has_solution_comment"].fillna(0).astype(int)
    df["unique_commenters"]   = df["unique_commenters"].fillna(0).astype(int)
    df["reassignment_count"]  = df["reassignment_count"].fillna(0).astype(int)
    df["reopen_count"]        = df["reopen_count"].fillna(0).astype(int)
    df["unique_actors"]       = df["unique_actors"].fillna(0).astype(int)

    # ── turn_count ──────────────────────────────────────────────────────
    # One turn = one comment round-trip; proxy = comment_count
    df["turn_count"] = df["comment_count"]

    # ── resolution_success ──────────────────────────────────────────────
    # Closed issue WITH a solution comment is a resolved success
    df["resolution_success"] = (
        (df["state"] == "closed") & (df["has_solution_comment"] == 1)
    ).astype(int)

    # ── SLA flags ───────────────────────────────────────────────────────
    df["sla_limit_hours"] = df["primary_label"].apply(get_sla_limit)
    df["sla_breach_flag"] = (
        df["resolution_time_hours"] > df["sla_limit_hours"]
    ).astype(int)

    df["sla_remaining_norm"] = (
        (df["sla_limit_hours"] - df["resolution_time_hours"]) / df["sla_limit_hours"]
    ).clip(0, 1).round(4)

    # ── frustration_score ───────────────────────────────────────────────
    # Higher turns + reassignments + reopens → higher frustration
    # Formula: normalised to [0, 1] range approximately
    raw_frustration = (
        df["turn_count"] * 0.5
        + df["reassignment_count"] * 1.5
        + df["reopen_count"] * 2.0
    )
    max_f = raw_frustration.max() if raw_frustration.max() > 0 else 1
    df["frustration_score"] = (raw_frustration / max_f).round(4)

    # ── interaction_depth ───────────────────────────────────────────────
    df["interaction_depth"] = df["unique_commenters"] + df["reassignment_count"]

    # ── first_response_time ─────────────────────────────────────────────
    # Placeholder — requires first comment timestamp, filled in merge below
    df["first_response_time_hours"] = np.nan

    rl_cols = [
        "issue_number",
        "turn_count", "reassignment_count", "reopen_count",
        "resolution_success", "sla_breach_flag", "sla_remaining_norm",
        "sla_limit_hours", "frustration_score", "interaction_depth",
        "first_response_time_hours",
    ]
    return df[rl_cols]


# ═══════════════════════════════════════════════════════════
# NLP FEATURES
# ═══════════════════════════════════════════════════════════

def compute_nlp_features(issues: pd.DataFrame,
                         agg_comments: pd.DataFrame) -> pd.DataFrame:
    """
    Computes NLP / RAG-ready features.

    Features:
        text_length          — char length of clean_text
        word_count           — word count of clean_text
        missing_version_flag — already in cleaned_issues, propagated
        missing_error_flag   — already in cleaned_issues, propagated
        has_solution_comment — from comments aggregation
        all_comments_text    — aggregated comment text (for RAG)
        solution_comments    — solution-only comments text (RAG knowledge base)
        urgency_keyword_flag — if urgency words present in text
        question_mark_flag   — if issue contains explicit question
        code_block_flag      — if original body had code blocks
    """
    df = issues.copy()
    # Drop the placeholder column from clean_issues to avoid collision
    if "has_solution_comment" in df.columns:
        df = df.drop(columns=["has_solution_comment"])

    # Merge comments
    df = df.merge(
        agg_comments[["issue_number", "has_solution_comment",
                       "all_comments_text", "solution_comments", "comment_count"]],
        on="issue_number", how="left"
    )
    df["all_comments_text"]    = df["all_comments_text"].fillna("")
    df["solution_comments"]    = df["solution_comments"].fillna("")
    df["has_solution_comment"] = df["has_solution_comment"].fillna(0).astype(int)
    df["comment_count"]        = df["comment_count"].fillna(0).astype(int)

    # ── Text stats ──────────────────────────────────────────────────────
    df["text_length"] = df["clean_text"].str.len()
    df["word_count"]  = df["clean_text"].str.split().str.len().fillna(0).astype(int)

    # ── Urgency keywords ────────────────────────────────────────────────
    _URGENCY_RE = re.compile(
        r"\b(urgent|critical|blocker|block(?:ing|ed)|crash(?:ing|es|ed)?|"
        r"broken|severe|asap|immediately|cannot|can\'t|unable)\b", re.IGNORECASE
    )
    df["urgency_keyword_flag"] = df["clean_text"].apply(
        lambda t: int(bool(_URGENCY_RE.search(str(t))))
    )

    # ── Question flag ───────────────────────────────────────────────────
    df["question_mark_flag"] = df["clean_text"].str.contains(r"\?", regex=True).astype(int)

    # ── Code block flag ─────────────────────────────────────────────────
    df["code_block_flag"] = df["clean_text"].str.contains(
        r"\[code_block\]", case=False, regex=True
    ).astype(int)

    nlp_cols = [
        "issue_number",
        "clean_text", "all_comments_text", "solution_comments",
        "text_length", "word_count",
        "missing_version_flag", "missing_error_flag",
        "has_solution_comment",
        "urgency_keyword_flag", "question_mark_flag", "code_block_flag",
    ]
    return df[nlp_cols]


# ═══════════════════════════════════════════════════════════
# FIRST RESPONSE TIME (from cleaned comments with timestamps)
# ═══════════════════════════════════════════════════════════

def compute_first_response_time(issues: pd.DataFrame,
                                comments_detail: pd.DataFrame) -> pd.Series:
    """Returns Series[issue_number → first_response_time_hours]."""
    first_comment = (
        comments_detail.sort_values("comment_created_at")
        .groupby("issue_number")
        .first()
        .reset_index()[["issue_number", "comment_created_at"]]
        .rename(columns={"comment_created_at": "first_comment_at"})
    )
    merged = issues[["issue_number", "created_at"]].merge(first_comment, on="issue_number", how="left")
    merged["created_at"]     = pd.to_datetime(merged["created_at"],     utc=True, errors="coerce")
    merged["first_comment_at"] = pd.to_datetime(merged["first_comment_at"], utc=True, errors="coerce")
    diff = (merged["first_comment_at"] - merged["created_at"]).dt.total_seconds() / 3600
    return diff.clip(lower=0).round(2).rename("first_response_time_hours")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def run(issues_path   = "data/intermediate/cleaned_issues.csv",
        comments_path = "data/intermediate/aggregated_comments.csv",
        events_path   = "data/intermediate/aggregated_events.csv",
        comments_detail_path = "data/intermediate/cleaned_comments.csv",
        nlp_out       = "data/intermediate/features_nlp.csv",
        rl_out        = "data/intermediate/features_rl.csv",
        final_out     = "data/final/final_dataset.csv"):

    print("[feature_engineering] Loading data ...")
    issues   = pd.read_csv(issues_path)
    agg_com  = pd.read_csv(comments_path)
    agg_ev   = pd.read_csv(events_path)
    com_det  = pd.read_csv(comments_detail_path)

    # ── NLP features ────────────────────────────────────────────────────
    print("  Computing NLP features ...")
    nlp_df = compute_nlp_features(issues, agg_com)
    nlp_df.to_csv(nlp_out, index=False)
    print(f"  Saved NLP features → {nlp_out}")

    # ── RL features ─────────────────────────────────────────────────────
    print("  Computing RL features ...")
    rl_df = compute_rl_features(issues, agg_com, agg_ev)

    # Fill first_response_time_hours from detailed comments
    frt = compute_first_response_time(issues, com_det)
    issues["_frt"] = frt.values
    rl_df = rl_df.merge(issues[["issue_number", "_frt"]], on="issue_number", how="left")
    rl_df["first_response_time_hours"] = rl_df["_frt"].fillna(-1).round(2)
    rl_df = rl_df.drop(columns=["_frt"])

    rl_df.to_csv(rl_out, index=False)
    print(f"  Saved RL features → {rl_out}")

    # ── Final merged dataset ─────────────────────────────────────────────
    print("  Building final merged dataset ...")
    final = issues.merge(nlp_df.drop(columns=["clean_text"]), on="issue_number", how="left")
    final = final.merge(rl_df, on="issue_number", how="left")

    final.to_csv(final_out, index=False)
    print(f"\n[feature_engineering] Final dataset → {final_out}  ({len(final)} rows, {len(final.columns)} cols)")
    print(f"  Columns: {list(final.columns)}")
    return final


if __name__ == "__main__":
    run()