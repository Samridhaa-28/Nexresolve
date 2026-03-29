"""
state_builder.py — RL State Vector Constructor
================================================
Reads the precomputed NLP features from final_rl_dataset.csv, merges the new
RAG signals (max_sim, avg_sim, sim_spread, knowledge_gap_flag), and outputs a
flat numeric state vector per ticket suitable for feeding into the RL agent.

Design principles:
  - NEVER recomputes NLP features — they are read as-is from the CSV.
  - RAG signals are computed here OR pre-computed and passed in as a dict.
  - Categorical columns (e.g. intent_group, sentiment_label) are label-encoded
    using a consistent, deterministic mapping so the vector is fully numeric.
  - The final state vector order is FIXED and documented in STATE_COLUMNS below.

State vector structure (39 numeric features total):
  ┌─────────────────────────── Existing NLP features (35) ─────────────────┐
  │ ticket_meta (5): text_length, word_count, question_mark_flag,           │
  │                  has_solution_comment, turn_count                        │
  │ sla (4): sla_breach_flag, sla_remaining_norm, sla_limit_hours,          │
  │          interaction_depth                                               │
  │ intent (4): intent_group_enc, confidence_score, uncertainty_flag,       │
  │             confidence_band_enc                                          │
  │ urgency (2): urgency_score, urgent_flag                                  │
  │ entity (8): has_version, has_error_type, has_platform, has_hardware,    │
  │             entity_count, missing_count, completeness_score              │
  │             reassignment_count                                            │
  │ clarification (2): needs_clarification, clarification_priority          │
  │ sentiment (2): sentiment_score, sentiment_label_enc                     │
  │ frustration (1): frustration_level                                      │
  │ routing (3): rl_recommendation_enc, reopen_count, resolution_success    │
  └─────────────────────────── New RAG signals (4) ─────────────────────────┐
    max_sim, avg_sim, sim_spread, knowledge_gap_flag                         │
  └─────────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

from nlp.nlp_pipeline import run_rag_for_dataframe, run_rag_for_ticket

# ---------------------------------------------------------------------------
# Deterministic label encodings (must be FIXED — never fit on test data)
# ---------------------------------------------------------------------------
INTENT_GROUP_MAP = {
    "duplicate"       : 0,
    "needs_info"      : 1,
    "bug"             : 2,
    "billing"         : 3,
    "feature_request" : 4,
    "general"         : 5,
    "triage"          : 6,
    "other"           : 7,
}

CONFIDENCE_BAND_MAP = {
    "low"    : 0,
    "medium" : 1,
    "high"   : 2,
}

SENTIMENT_LABEL_MAP = {
    "negative" : 0,
    "neutral"  : 1,
    "positive" : 2,
}

RL_RECOMMENDATION_MAP = {
    "clarify_first"    : 0,
    "route_or_clarify" : 1,
    "route_directly"   : 2,
    "escalate"         : 3,
}

# ---------------------------------------------------------------------------
# Ordered state columns (the exact order of the output vector)
# ---------------------------------------------------------------------------
STATE_COLUMNS = [
    # ── Ticket meta ──────────────────────────────────────────────────────
    "text_length",
    "word_count",
    "question_mark_flag",
    "has_solution_comment",
    "turn_count",
    # ── SLA ──────────────────────────────────────────────────────────────
    "sla_breach_flag",
    "sla_remaining_norm",
    "sla_limit_hours",
    "interaction_depth",
    # ── Intent ───────────────────────────────────────────────────────────
    "intent_group_enc",
    "confidence_score",
    "uncertainty_flag",
    "confidence_band_enc",
    # ── Urgency ──────────────────────────────────────────────────────────
    "urgency_score",
    "urgent_flag",
    # ── Entity extraction ─────────────────────────────────────────────────
    "has_version",
    "has_error_type",
    "has_platform",
    "has_hardware",
    "entity_count",
    "missing_count",
    "completeness_score",
    "reassignment_count",
    # ── Clarification ─────────────────────────────────────────────────────
    "needs_clarification",
    "clarification_priority",
    # ── Sentiment / frustration ───────────────────────────────────────────
    "sentiment_score",
    "sentiment_label_enc",
    "frustration_level",
    # ── Routing ───────────────────────────────────────────────────────────
    "rl_recommendation_enc",
    "reopen_count",
    "resolution_success",
    # ── RAG signals (new) ─────────────────────────────────────────────────
    "max_sim",
    "avg_sim",
    "sim_spread",
    "knowledge_gap_flag",
    "tier1_flag",
    "tier2_flag",
]

# Total expected state dimension
STATE_DIM = len(STATE_COLUMNS)   # 35


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

def _encode_column(series: pd.Series, mapping: dict, default: int = 7) -> pd.Series:
    """Map string labels to integers; unknown values get `default`."""
    return series.str.lower().str.strip().map(mapping).fillna(default).astype(int)


def _apply_encodings(df: pd.DataFrame) -> pd.DataFrame:
    """Add encoded columns in-place and return df."""
    df["intent_group_enc"]       = _encode_column(df["intent_group"],       INTENT_GROUP_MAP)
    df["confidence_band_enc"]    = _encode_column(df["confidence_band"],     CONFIDENCE_BAND_MAP, default=1)
    df["sentiment_label_enc"]    = _encode_column(df["sentiment_label"],     SENTIMENT_LABEL_MAP, default=1)
    df["rl_recommendation_enc"]  = _encode_column(df["rl_recommendation"],   RL_RECOMMENDATION_MAP, default=1)
    
    # ── Tier flags (from top_tier string) ──
    if "top_tier" in df.columns:
        df["tier1_flag"] = (df["top_tier"] == "tier1_verified").astype(int)
        df["tier2_flag"] = (df["top_tier"] == "tier2_discussed").astype(int)
    else:
        df["tier1_flag"] = 0
        df["tier2_flag"] = 0
        
    return df


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------

def build_state_vectors(
    dataset_path: str  = "data/final/final_rl_dataset.csv",
    index_path:   str  = "data/retrieval/kb_index",
    output_path:  str  = "data/final/rl_ready_dataset.csv",
    top_k:        int  = 5,
    save:         bool = True,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Full pipeline: load NLP dataset → add RAG signals → encode → build state matrix.

    Parameters
    ----------
    dataset_path : path to final_rl_dataset.csv
    index_path   : FAISS index path prefix
    output_path  : where to write the RAG-augmented CSV
    top_k        : top-k retrieved issues per ticket
    save         : write output CSV to disk

    Returns
    -------
    (df_augmented, state_matrix)
      df_augmented  : DataFrame with all original columns + 4 RAG columns
      state_matrix  : np.ndarray of shape (n_tickets, STATE_DIM), dtype float32
    """
    # ── 1. Load existing NLP dataset ──────────────────────────────────────
    print(f"[StateBuilder] Loading dataset: {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f"[StateBuilder] {len(df)} tickets loaded, {df.shape[1]} existing features.")

    # ── 2. Add RAG signals ────────────────────────────────────────────────
    print("[StateBuilder] Running RAG pipeline …")
    df = run_rag_for_dataframe(
        df,
        text_col          = "clean_text",
        intent_col        = "intent_group",
        confidence_col    = "confidence_score",
        top_k             = top_k,
        index_path        = index_path,
        show_progress     = True,
    )
    print("[StateBuilder] RAG signals added: max_sim, avg_sim, sim_spread, knowledge_gap_flag.")
    


    # ── 3. Apply label encodings ──────────────────────────────────────────
    df = _apply_encodings(df)

    # ── 4. Build numeric state matrix ─────────────────────────────────────
    missing = [c for c in STATE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"[StateBuilder] Missing columns in augmented dataset: {missing}\n"
            "Check that the source CSV and RAG pipeline are producing all expected features."
        )

    state_matrix = df[STATE_COLUMNS].astype(float).values.astype(np.float32)
    print(f"[StateBuilder] State matrix shape: {state_matrix.shape}  (tickets × features)")

    # ── 5. Validate for NaN / Inf ─────────────────────────────────────────
    bad_rows = np.where(~np.isfinite(state_matrix).all(axis=1))[0]
    if len(bad_rows):
        print(f"[StateBuilder] WARNING: {len(bad_rows)} rows have NaN/Inf — filling with 0.")
        state_matrix = np.nan_to_num(state_matrix, nan=0.0, posinf=1.0, neginf=0.0)

    # ── 6. Save augmented CSV ─────────────────────────────────────────────
    if save:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"[StateBuilder] Saved augmented dataset → {output_path}")

    return df, state_matrix


def build_single_state(
    ticket_row: dict | pd.Series,
    index_path: str = "data/retrieval/kb_index",
    top_k:      int = 5,
) -> np.ndarray:
    """
    Build a state vector for a SINGLE ticket at inference time.

    Parameters
    ----------
    ticket_row : dict or Series with all NLP feature keys
    index_path : FAISS index path prefix
    top_k      : top-k retrieval candidates

    Returns
    -------
    np.ndarray of shape (STATE_DIM,), dtype float32
    """
    row = dict(ticket_row)

    # RAG signals
    rag = run_rag_for_ticket(
        clean_text        = str(row.get("clean_text", "")),
        intent_label      = str(row.get("intent_group", "")),
        intent_confidence = float(row.get("confidence_score", 1.0)),
        top_k             = top_k,
        index_path        = index_path,
    )
    row.update(rag)

    # Encode categoricals
    row["intent_group_enc"]      = INTENT_GROUP_MAP      .get(str(row.get("intent_group", "")).lower(), 7)
    row["confidence_band_enc"]   = CONFIDENCE_BAND_MAP   .get(str(row.get("confidence_band", "")).lower(), 1)
    row["sentiment_label_enc"]   = SENTIMENT_LABEL_MAP   .get(str(row.get("sentiment_label", "")).lower(), 1)
    row["rl_recommendation_enc"] = RL_RECOMMENDATION_MAP .get(str(row.get("rl_recommendation", "")).lower(), 1)

    # Tier flags
    top_tier = str(row.get("top_tier", "tier3_minimal"))
    row["tier1_flag"] = 1 if top_tier == "tier1_verified" else 0
    row["tier2_flag"] = 1 if top_tier == "tier2_discussed" else 0

    # Build vector
    vec = np.array([float(row.get(c, 0.0)) for c in STATE_COLUMNS], dtype=np.float32)
    return vec


def get_state_column_names() -> list[str]:
    """Return the ordered list of feature names in the state vector."""
    return list(STATE_COLUMNS)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== RL State Builder Demo ===\n")

    # Single-ticket demo
    sample_row = {
    "clean_text"          : "login fails after password reset, invalid session error",

    # ── Ticket meta ─────────────────────────
    "text_length"         : 62,
    "word_count"          : 10,
    "question_mark_flag"  : 0,
    "has_solution_comment": 0,
    "turn_count"          : 1,

    # ── SLA ────────────────────────────────
    "sla_breach_flag"     : 0,
    "sla_remaining_norm"  : 0.80,
    "sla_limit_hours"     : 24,
    "interaction_depth"   : 0,

    # ── Intent ─────────────────────────────
    "intent_group"        : "bug",
    "confidence_score"    : 0.78,
    "uncertainty_flag"    : 0,
    "confidence_band"     : "medium",

    # ── Urgency ────────────────────────────
    "urgency_score"       : 0.55,
    "urgent_flag"         : 1,

    # ── Entity ─────────────────────────────
    "has_version"         : 0,
    "has_error_type"      : 1,
    "has_platform"        : 0,
    "has_hardware"        : 0,
    "entity_count"        : 1,
    "missing_count"       : 3,
    "completeness_score"  : 0.25,
    "reassignment_count"  : 0,

    # ── Clarification ──────────────────────
    "needs_clarification" : 1 ,
    "clarification_priority": 2,

    # ── Sentiment ──────────────────────────
    "sentiment_score"     : -0.45,
    "sentiment_label"     : "negative",
    "frustration_level"   : 0.72,

    # ── Routing ────────────────────────────
    "rl_recommendation"   : "clarify_first",
    "reopen_count"        : 0,
    "resolution_success"  : 0,

    # ── NEW: RAG features (dummy values for demo) ─────────
    "max_sim"             : 0.82,
    "avg_sim"             : 0.76,
    "sim_spread"          : 0.12,
    "knowledge_gap_flag"  : 0,

    # ── NEW: Tier info ─────────────────────
    "top_tier"            : "tier1_verified",

    # ── NEW: Tier flags ────────────────────
    "tier1_flag"          : 1,
    "tier2_flag"          : 0,
}

    vec = build_single_state(sample_row)
    print(f"State vector shape : {vec.shape}")
    print(f"State vector dtype : {vec.dtype}")
    print(f"\nFeature names ({STATE_DIM}):")
    for name, val in zip(STATE_COLUMNS, vec):
        print(f"  {name:<28} = {val:.4f}")
