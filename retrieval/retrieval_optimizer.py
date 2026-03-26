from __future__ import annotations

import re
from typing import Optional


# ---------------------------------------------------------------------------
# Tuneable parameters (can be overridden per call)
# ---------------------------------------------------------------------------
DEFAULT_SIM_THRESHOLD      = 0.50    # minimum cosine similarity to keep
DEFAULT_DEDUP_JACCARD_THRESH = 0.70  # solutions this similar are treated as duplicates


# ---------------------------------------------------------------------------
# 1. Intent-aware filtering
# ---------------------------------------------------------------------------

def filter_by_intent(
    retrieved: list[dict],
    intent_label: Optional[str] = None,
    confidence: float = 1.0,
) -> list[dict]:
    
    if not intent_label or confidence < 0.50:
        return retrieved

    intent_lower = intent_label.lower()
    filtered = [
        r for r in retrieved
        if intent_lower in r.get("primary_label", "").lower()
        or r.get("primary_label", "").lower() in intent_lower
    ]

    # Fallback: never return an empty list
    return filtered if filtered else retrieved


# ---------------------------------------------------------------------------
# 2. Template deduplication
# ---------------------------------------------------------------------------

def _jaccard(text_a: str, text_b: str) -> float:
    """Jaccard similarity between word-sets of two strings."""
    words_a = set(re.findall(r"\w+", text_a.lower()))
    words_b = set(re.findall(r"\w+", text_b.lower()))
    if not words_a and not words_b:
        return 1.0
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


def deduplicate_solutions(
    retrieved: list[dict],
    jaccard_threshold: float = DEFAULT_DEDUP_JACCARD_THRESH,
) -> list[dict]:

    kept: list[dict] = []
    kept_solutions: list[str] = []

    # NEW: track duplicate count per cluster
    for result in retrieved:
        candidate_sol = result.get("solution_comments", "")

        is_duplicate = False

        for i, kept_sol in enumerate(kept_solutions):
            if _jaccard(candidate_sol, kept_sol) >= jaccard_threshold:
                # increment duplicate count
                kept[i]["duplicate_count"] = kept[i].get("duplicate_count", 1) + 1
                is_duplicate = True
                break

        if not is_duplicate:
            result["duplicate_count"] = 1   # first occurrence
            kept.append(result)
            kept_solutions.append(candidate_sol)

    return kept if kept else retrieved[:1]


# ---------------------------------------------------------------------------
# 3. Similarity thresholding
# ---------------------------------------------------------------------------

def apply_threshold(
    retrieved: list[dict],
    threshold: float = DEFAULT_SIM_THRESHOLD,
) -> list[dict]:
   
    above = [r for r in retrieved if r["similarity_score"] >= threshold]
    if not above and retrieved:
        return retrieved[:1]    # fallback: keep best even if below threshold
    return above


# ---------------------------------------------------------------------------
# Composite optimiser
# ---------------------------------------------------------------------------

def optimize_retrieval(
    retrieved: list[dict],
    intent_label: Optional[str] = None,
    intent_confidence: float = 1.0,
    sim_threshold: float = DEFAULT_SIM_THRESHOLD,
    jaccard_threshold: float = DEFAULT_DEDUP_JACCARD_THRESH,
) -> list[dict]:
    """
    Apply all three optimisation steps in sequence.

    Parameters
    ----------
    retrieved         : raw results from retrieve_similar_issues()
    intent_label      : predicted intent of the incoming ticket (may be None)
    intent_confidence : confidence of the intent prediction (0–1)
    sim_threshold     : minimum similarity score to keep
    jaccard_threshold : solution Jaccard threshold for deduplication

    Returns
    -------
    Optimised list of result dicts (same schema as input)
    """
    if not retrieved:
        return []

    # Step 1 — intent filter
    step1 = filter_by_intent(retrieved, intent_label, intent_confidence)

    # Step 2 — dedup
    step2 = deduplicate_solutions(step1, jaccard_threshold)

    # Step 3 — threshold
    step3 = apply_threshold(step2, sim_threshold)

    return step3


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import pprint

    raw = [
        {"rank": 1, "issue_number": 1, "primary_label": "billing",
         "similarity_score": 0.88, "solution_comments": "clear cache and restart billing portal",
         "clean_text": "billing error"},
        {"rank": 2, "issue_number": 2, "primary_label": "duplicate",
         "similarity_score": 0.80, "solution_comments": "this is a duplicate issue please close",
         "clean_text": "billing problem"},
        {"rank": 3, "issue_number": 3, "primary_label": "billing",
         "similarity_score": 0.77, "solution_comments": "clear the cache and restart billing portal",  # near-dup of rank 1
         "clean_text": "payment not processing"},
        {"rank": 4, "issue_number": 4, "primary_label": "billing",
         "similarity_score": 0.40, "solution_comments": "contact support",
         "clean_text": "subscription issue"},
    ]

    optimised = optimize_retrieval(raw, intent_label="billing", intent_confidence=0.85)
    print("=== Optimised Retrieval ===")
    pprint.pprint(optimised)
