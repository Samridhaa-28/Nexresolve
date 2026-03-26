import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_GAP_THRESHOLD = 0.6 # max_sim below this → knowledge gap detected
HIGH_CONFIDENCE_THRESHOLD = 0.85  # max_sim at or above this → strong evidence


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def compute_knowledge_gap_flag(
    retrieved: list[dict],
    gap_threshold: float = DEFAULT_GAP_THRESHOLD,
) -> int:
    
    if not retrieved:
        return 1   # no results at all → always a gap

    max_sim = max(r["similarity_score"] for r in retrieved)
    return 0 if max_sim >= gap_threshold else 1


def compute_retrieval_confidence(
    retrieved: list[dict],
    gap_threshold: float = DEFAULT_GAP_THRESHOLD,
    high_threshold: float = HIGH_CONFIDENCE_THRESHOLD,
) -> dict:
    """
    Extended gap analysis returning both the binary flag and a graded
    confidence level label.

    Parameters
    ----------
    retrieved      : raw or optimised retrieval results
    gap_threshold  : max_sim cutoff for gap flag
    high_threshold : max_sim cutoff for "high" confidence

    Returns
    -------
    dict with:
        knowledge_gap_flag : int  — 0 or 1
        retrieval_confidence_level : str — "high" | "moderate" | "low"
        max_sim_seen       : float
    """
    if not retrieved:
        return {
            "knowledge_gap_flag"         : 1,
            "retrieval_confidence_level" : "low",
            "max_sim_seen"               : 0.0,
        }

    max_sim = float(max(r["similarity_score"] for r in retrieved))
    gap_flag = 0 if max_sim >= gap_threshold else 1

    if max_sim >= high_threshold:
        confidence_level = "high"
    elif max_sim >= gap_threshold:
        confidence_level = "moderate"
    else:
        confidence_level = "low"

    return {
        "knowledge_gap_flag"         : gap_flag,
        "retrieval_confidence_level" : confidence_level,
        "max_sim_seen"               : round(max_sim, 4),
    }


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Scenario A: good retrieval
    good_results = [
        {"similarity_score": 0.82},
        {"similarity_score": 0.78},
        {"similarity_score": 0.71},
    ]
    print("Scenario A (good):")
    print(" ", compute_retrieval_confidence(good_results))

    # Scenario B: poor retrieval
    poor_results = [
        {"similarity_score": 0.42},
        {"similarity_score": 0.38},
    ]
    print("\nScenario B (poor):")
    print(" ", compute_retrieval_confidence(poor_results))

    # Scenario C: no results
    print("\nScenario C (no results):")
    print(" ", compute_retrieval_confidence([]))
