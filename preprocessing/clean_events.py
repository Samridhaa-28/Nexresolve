"""
NexResolve — Step 3: Clean Events
Input : data/raw/events.csv
Output: data/intermediate/cleaned_events.csv
        data/intermediate/aggregated_events.csv  (one row per issue)

Run: python preprocessing/clean_events.py
"""

import pandas as pd


# ─────────────────────────────────────────────
# Useful event types (signal-bearing only)
# ─────────────────────────────────────────────
USEFUL_EVENTS = {
    "assigned", "unassigned", "labeled", "unlabeled",
    "closed", "reopened", "mentioned", "referenced",
    "milestoned", "demilestoned", "renamed",
}


def run(input_path: str = "data/raw/events.csv",
        output_path: str = "data/intermediate/cleaned_events.csv",
        agg_path: str    = "data/intermediate/aggregated_events.csv"):

    print(f"[clean_events] Loading {input_path} ...")
    df = pd.read_csv(input_path)
    print(f"  Raw rows: {len(df)}")

    # ── 1. Keep only useful event types ─────────────────────────────────
    df = df[df["event_type"].isin(USEFUL_EVENTS)].copy()
    print(f"  After filtering event types: {len(df)} rows")

    # ── 2. Parse timestamps ──────────────────────────────────────────────
    df["event_created_at"] = pd.to_datetime(
        df["event_created_at"], errors="coerce", utc=True
    )

    # ── 3. Clean string fields ───────────────────────────────────────────
    df["actor_login"]    = df["actor_login"].fillna("unknown")
    df["label_name"]     = df["label_name"].fillna("")
    df["assignee_login"] = df["assignee_login"].fillna("")

    # ── 4. Bot actor filter ──────────────────────────────────────────────
    df["is_bot_actor"] = df["actor_login"].str.lower().str.contains(r"\[bot\]|bot$", regex=True)

    df.to_csv(output_path, index=False)
    print(f"[clean_events] Saved cleaned events → {output_path}")

    # ─────────────────────────────────────────────────────────────────────
    # AGGREGATION: RL-useful signals per issue
    # ─────────────────────────────────────────────────────────────────────
    assigned_df    = df[df["event_type"] == "assigned"]
    unassigned_df  = df[df["event_type"] == "unassigned"]
    reopened_df    = df[df["event_type"] == "reopened"]
    labeled_df     = df[df["event_type"] == "labeled"]

    agg = df.groupby("issue_number").agg(
        total_events       = ("event_id",    "count"),
        reopen_count       = ("event_type",  lambda x: (x == "reopened").sum()),
        unique_actors      = ("actor_login", "nunique"),
    ).reset_index()

    # reassignment_count = number of "assigned" events per issue
    assign_counts = assigned_df.groupby("issue_number").size().rename("reassignment_count")
    agg = agg.merge(assign_counts, on="issue_number", how="left")
    agg["reassignment_count"] = agg["reassignment_count"].fillna(0).astype(int)

    # first_assignee
    first_assignee = (
        assigned_df.sort_values("event_created_at")
        .groupby("issue_number")["assignee_login"]
        .first()
        .rename("first_assignee")
    )
    agg = agg.merge(first_assignee, on="issue_number", how="left")

    # sla_breach_flag — will be properly computed in feature_engineering.py
    # but we add a placeholder here
    agg["sla_breach_flag"] = 0

    # all_labels_seen — useful for NLP label distribution analysis
    labels_seen = (
        labeled_df.groupby("issue_number")["label_name"]
        .apply(lambda x: "|".join(x.dropna().unique()))
        .rename("all_labels_seen")
    )
    agg = agg.merge(labels_seen, on="issue_number", how="left")
    agg["all_labels_seen"] = agg["all_labels_seen"].fillna("")

    agg.to_csv(agg_path, index=False)
    print(f"[clean_events] Saved aggregated events → {agg_path}  ({len(agg)} issues)")
    return df, agg


if __name__ == "__main__":
    import sys
    run(
        input_path  = sys.argv[1] if len(sys.argv) > 1 else "data/raw/events.csv",
        output_path = sys.argv[2] if len(sys.argv) > 2 else "data/intermediate/cleaned_events.csv",
        agg_path    = sys.argv[3] if len(sys.argv) > 3 else "data/intermediate/aggregated_events.csv",
    )