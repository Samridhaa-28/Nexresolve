
from __future__ import annotations

import os
from typing import Optional

from retrieval.faiss_index         import load_index, DEFAULT_INDEX_PATH
from retrieval.retriever           import retrieve_similar_issues, _clean_text
from retrieval.retrieval_optimizer import optimize_retrieval
from retrieval.similarity_features import compute_similarity_features
from retrieval.knowledge_gap       import compute_knowledge_gap_flag
from retrieval.embedder            import generate_embeddings

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_TOP_K          = 5
DEFAULT_GAP_THRESHOLD  = 0.6
DEFAULT_SIM_THRESHOLD  = 0.50
DEFAULT_JACC_THRESHOLD = 0.70


# ---------------------------------------------------------------------------
# Lazy-loaded index singleton
# ---------------------------------------------------------------------------
_index    = None
_metadata = None


def _get_index(index_path: str = DEFAULT_INDEX_PATH):
    """Load the FAISS index from disk once and cache in module globals."""
    global _index, _metadata
    if _index is None or _metadata is None:
        _index, _metadata = load_index(index_path)
    return _index, _metadata


def reset_index_cache() -> None:
    """Force reload of the FAISS index on next call (useful for testing)."""
    global _index, _metadata
    _index = None
    _metadata = None


# ---------------------------------------------------------------------------
# Core: process a single ticket
# ---------------------------------------------------------------------------

def run_rag_for_ticket(
    clean_text:       str,
    issue_number: int = None,
    intent_label:     Optional[str] = None,
    intent_confidence: float = 1.0,
    top_k:            int   = DEFAULT_TOP_K,
    gap_threshold:    float = DEFAULT_GAP_THRESHOLD,
    sim_threshold:    float = DEFAULT_SIM_THRESHOLD,
    jacc_threshold:   float = DEFAULT_JACC_THRESHOLD,
    index_path:       str   = DEFAULT_INDEX_PATH,
    return_full:      bool  = False,
) -> dict:
    """
    Run the full RAG pipeline for a single incoming ticket.

    Parameters
    ----------
    clean_text        : pre-cleaned ticket body (from final_rl_dataset.csv)
    intent_label      : predicted intent (e.g. "duplicate", "billing") — optional
    intent_confidence : model confidence in intent prediction (0–1)
    top_k             : how many KB candidates to retrieve
    gap_threshold     : max_sim below this → knowledge_gap_flag = 1
    sim_threshold     : minimum similarity to keep after optimisation
    jacc_threshold    : Jaccard threshold for solution deduplication
    index_path        : path prefix to .faiss / .meta.json files
    return_full       : if True, also include raw retrieval results in output

    Returns
    -------
    dict with at minimum these 4 RL signal keys:
        {
          "max_sim"             : float,
          "avg_sim"             : float,
          "sim_spread"          : float,
          "knowledge_gap_flag"  : int,
        }
    Plus (when return_full=True):
        {
          "retrieved_raw"       : list[dict],  # before optimisation
          "retrieved_optimised" : list[dict],  # after optimisation
          "n_retrieved"         : int,
        }
    """
    # ── Guard: empty text ─────────────────────────────────────────────────
    if not clean_text or not clean_text.strip():
        return _zero_rag_signals(return_full)

    # ── 1. Load / reuse index ────────────────────────────────────────────
    index, metadata = _get_index(index_path)

    # ── 2. Embed and retrieve ────────────────────────────────────────────
    query = _clean_text(clean_text)
    retrieved_raw = retrieve_similar_issues(
        query_text=query,
        index=index,
        metadata=metadata,
        top_k=top_k,
        query_issue_number=issue_number
    )

    # ── 3. Optimise retrieval ────────────────────────────────────────────
    retrieved_opt = optimize_retrieval(
        retrieved=retrieved_raw,
        intent_label=intent_label,
        intent_confidence=intent_confidence,
        sim_threshold=sim_threshold,
        jaccard_threshold=jacc_threshold,
    )
    
    top_tier = "tier3_minimal"
    if retrieved_opt:
        top_tier = retrieved_opt[0].get("kb_quality_tier", "tier3_minimal")

    # ── 4. Compute similarity features ───────────────────────────────────
    sim_features = compute_similarity_features(retrieved_opt)

    # ── 5. Knowledge gap flag ─────────────────────────────────────────────
    gap_flag = compute_knowledge_gap_flag(retrieved_opt, gap_threshold=gap_threshold)

    # ── 6. Assemble output ───────────────────────────────────────────────
    rag_signals = {
        "max_sim"            : sim_features["max_sim"],
        "avg_sim"            : sim_features["avg_sim"],
        "sim_spread"         : sim_features["sim_spread"],
        "knowledge_gap_flag" : gap_flag,
        "top_tier": top_tier 
    }

    if return_full:
        rag_signals["retrieved_raw"]       = retrieved_raw
        rag_signals["retrieved_optimised"] = retrieved_opt
        rag_signals["n_retrieved"]         = len(retrieved_opt)

    return rag_signals


# ---------------------------------------------------------------------------
# Batch: process an entire DataFrame
# ---------------------------------------------------------------------------

def run_rag_for_dataframe(
    df,
    text_col:          str   = "clean_text",
    intent_col:        str   = "intent_group",
    confidence_col:    str   = "confidence_score",
    top_k:             int   = DEFAULT_TOP_K,
    gap_threshold:     float = DEFAULT_GAP_THRESHOLD,
    sim_threshold:     float = DEFAULT_SIM_THRESHOLD,
    jacc_threshold:    float = DEFAULT_JACC_THRESHOLD,
    index_path:        str   = DEFAULT_INDEX_PATH,
    show_progress:     bool  = True,
):
    """
    Vectorised batch processing over a pandas DataFrame.

    Adds 4 new columns to a COPY of `df`:
        max_sim, avg_sim, sim_spread, knowledge_gap_flag

    Parameters
    ----------
    df             : pandas DataFrame (typically final_rl_dataset.csv)
    text_col       : column name for ticket text
    intent_col     : column name for predicted intent (may be absent)
    confidence_col : column name for intent confidence (may be absent)
    top_k          : candidates per ticket
    gap_threshold  : knowledge gap threshold
    sim_threshold  : minimum similarity threshold
    jacc_threshold : Jaccard deduplication threshold
    index_path     : FAISS index path prefix
    show_progress  : print a progress counter every 100 tickets

    Returns
    -------
    DataFrame (copy) with 4 new RAG feature columns appended
    """
    import pandas as pd

    df_out = df.copy()
    results = []

    # Pre-load index once before the loop
    _get_index(index_path)

    has_intent     = intent_col     in df.columns
    has_confidence = confidence_col in df.columns

    for i, (_, row) in enumerate(df.iterrows()):
        if show_progress and (i % 100 == 0):
            print(f"  [RAG] Processing row {i}/{len(df)} …")

        text       = str(row.get(text_col, "")) if text_col in df.columns else ""
        intent     = str(row[intent_col])     if has_intent     else None
        confidence = float(row[confidence_col]) if has_confidence else 1.0

        rag = run_rag_for_ticket(
            clean_text=text,
            issue_number=int(row["issue_number"]), 
            intent_label=intent,
            intent_confidence=confidence,
            top_k=top_k,
            gap_threshold=gap_threshold,
            sim_threshold=sim_threshold,
            jacc_threshold=jacc_threshold,
            index_path=index_path,
        )
        results.append(rag)

    # Append 4 new columns
    df_out["max_sim"]            = [r["max_sim"]            for r in results]
    df_out["avg_sim"]            = [r["avg_sim"]            for r in results]
    df_out["sim_spread"]         = [r["sim_spread"]         for r in results]
    df_out["knowledge_gap_flag"] = [r["knowledge_gap_flag"] for r in results]
    df_out["top_tier"] = [r.get("top_tier", "tier3_minimal") for r in results]

    return df_out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _zero_rag_signals(return_full: bool) -> dict:
    """Return safe zero-valued RAG signals when input is empty."""
    out = {
        "max_sim"            : 0.0,
        "avg_sim"            : 0.0,
        "sim_spread"         : 0.0,
        "knowledge_gap_flag" : 1,
    }
    if return_full:
        out["retrieved_raw"]       = []
        out["retrieved_optimised"] = []
        out["n_retrieved"]         = 0
    return out


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample = (
        "jupyter notebook crashes immediately on startup after updating vscode "
        "extension. throws a kernel not found error."
    )

    print("=== NLP Pipeline — RAG Demo ===\n")
    result = run_rag_for_ticket(
        clean_text=sample,
        intent_label="bug",
        intent_confidence=0.82,
        return_full=True,
    )

    print(f"max_sim            : {result['max_sim']}")
    print(f"avg_sim            : {result['avg_sim']}")
    print(f"sim_spread         : {result['sim_spread']}")
    print(f"knowledge_gap_flag : {result['knowledge_gap_flag']}")
    print(f"n_retrieved        : {result.get('n_retrieved', '-')}")

    print("\nTop retrieved issues:")
    for r in result.get("retrieved_optimised", []):
        print(f"  Rank {r['rank']} | sim={r['similarity_score']:.4f} | "
              f"label={r['primary_label']} | #{r['issue_number']}")
        print(f"    ↳ {r['solution_comments'][:100]}…")


# ---------------------------------------------------------------------------
# Combined NLP pipeline — used by the FastAPI backend (api/pipeline.py)
# ---------------------------------------------------------------------------

def run_nlp_pipeline(ticket_text: str) -> dict:
    """
    Run the full NLP stack for a single raw ticket string.

    Calls: intent_classifier → entity_extractor → missing_detector →
           urgency_predictor → sentiment_analyzer → clarification_modeler

    Returns a flat dict consumed by build_single_state().
    All numeric values are Python-native (not numpy) for JSON safety.
    No exceptions propagate — each module falls back to safe defaults.
    """
    # Local imports to avoid circular dependencies and delay heavy loading
    import nlp.intent_classifier    as _intent
    import nlp.entity_extractor     as _entity
    import nlp.missing_detector     as _missing
    import nlp.urgency_predictor    as _urgency
    import nlp.clarification_modeler as _clarify
    from nlp.sentiment_analyzer import SentimentAnalyzer

    text = (ticket_text or "").strip()

    # ── 1. Intent classification ──────────────────────────────────────────────
    try:
        ir = _intent.predict(text)
        intent_group     = str(ir.get("intent_group", "other"))
        confidence_score = float(ir.get("confidence_score", 0.5))
        uncertainty_flag = int(ir.get("uncertainty_flag", 0))
    except Exception:
        intent_group, confidence_score, uncertainty_flag = "other", 0.5, 0

    # Confidence band (deterministic, matches CONFIDENCE_BAND_MAP in state_builder)
    if confidence_score < 0.4:
        confidence_band = "low"
    elif confidence_score < 0.7:
        confidence_band = "medium"
    else:
        confidence_band = "high"

    # ── 2. Entity extraction ──────────────────────────────────────────────────
    try:
        er    = _entity.predict(text)
        flags = er.get("flags", {})
        has_version    = int(flags.get("has_version",    0))
        has_error_type = int(flags.get("has_error_type", 0))
        has_platform   = int(flags.get("has_platform",   0))
        has_hardware   = int(flags.get("has_hardware",   0))
        entity_count   = int(flags.get("entity_count",   0))
    except Exception:
        has_version = has_error_type = has_platform = has_hardware = entity_count = 0

    # ── 3. Missing field detection ────────────────────────────────────────────
    try:
        mr = _missing.predict(
            intent_group  = intent_group,
            has_version   = has_version,
            has_error_type= has_error_type,
            has_platform  = has_platform,
            has_hardware  = has_hardware,
        )
        missing_fields    = mr.get("missing_fields", [])
        missing_count     = int(len(missing_fields))
        completeness_score = float(1.0 - missing_count / 4.0)
    except Exception:
        missing_count, completeness_score = 0, 1.0

    # ── 4. Urgency prediction ─────────────────────────────────────────────────
    try:
        ur = _urgency.predict(text)
        urgency_score = float(ur.get("urgency_score", 0.0))
        urgent_flag   = int(ur.get("urgent_flag",   0))
    except Exception:
        urgency_score, urgent_flag = 0.0, 0

    # ── 5. Sentiment analysis ─────────────────────────────────────────────────
    try:
        sa = SentimentAnalyzer(model="vader")
        sentiment_score = float(sa.analyze_text(text))
        sentiment_label = sa.get_label(sentiment_score)
    except Exception:
        sentiment_score, sentiment_label = 0.0, "neutral"

    # Frustration level: derived from negative sentiment (0–1 range)
    frustration_level = float(max(0.0, -sentiment_score))

    # Rule-based boost: scan for frustration keywords (+0.4 per match, capped at 1.0)
    _FRUSTRATION_KEYWORDS = {
        "irritated", "dissatisfied", "unhappy", "frustrated",
        "annoyed", "angry", "upset", "disappointed", "irritate", "fed up"
    }
    _text_lower = text.lower()
    for _kw in _FRUSTRATION_KEYWORDS:
        if _kw in _text_lower:
            frustration_level += 0.4
    frustration_level = min(1.0, frustration_level)

    # ── 6. Clarification modeling ─────────────────────────────────────────────
    try:
        cr = _clarify.predict(
            intent_group     = intent_group,
            uncertainty_flag = uncertainty_flag,
            missing_version  = 1 - has_version,
            missing_error    = 1 - has_error_type,
            missing_platform = 1 - has_platform,
            missing_hardware = 1 - has_hardware,
            word_count       = len(text.split()),
        )
        needs_clarification    = int(cr.get("needs_clarification",    0))
        clarification_priority = float(cr.get("clarification_priority", 0.0))
    except Exception:
        needs_clarification, clarification_priority = 0, 0.0

    # ── Assemble flat result dict ─────────────────────────────────────────────
    return {
        # Intent
        "intent_group":            intent_group,
        "confidence_score":        confidence_score,
        "uncertainty_flag":        uncertainty_flag,
        "confidence_band":         confidence_band,
        # Entity
        "has_version":             has_version,
        "has_error_type":          has_error_type,
        "has_platform":            has_platform,
        "has_hardware":            has_hardware,
        "entity_count":            entity_count,
        # Missing
        "missing_count":           missing_count,
        "completeness_score":      completeness_score,
        # Urgency
        "urgency_score":           urgency_score,
        "urgent_flag":             urgent_flag,
        # Sentiment
        "sentiment_score":         sentiment_score,
        "sentiment_label":         sentiment_label,
        "frustration_level":       frustration_level,
        # Clarification
        "needs_clarification":     needs_clarification,
        "clarification_priority":  clarification_priority,
    }
