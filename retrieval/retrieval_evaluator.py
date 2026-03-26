import numpy as np


# ---------------------------------------------------------------------------
# Quality report structure (returned as a plain dict for simplicity)
# ---------------------------------------------------------------------------

def evaluate_retrieval(
    retrieved: list[dict],
    query_label: str | None = None,
    relevant_threshold: float = 0.60,
) -> dict:
    """
    Assess the quality of a retrieved result set.

    Parameters
    ----------
    retrieved           : list of dicts from retrieve_similar_issues()
    query_label         : ground-truth intent/label of the query (optional)
    relevant_threshold  : minimum similarity score to count as "relevant"

    Returns
    -------
    dict with keys:
        n_retrieved         : int   — how many results were returned
        n_above_threshold   : int   — results with sim ≥ relevant_threshold
        top1_score          : float — best similarity score
        mean_score          : float — average of all retrieved similarities
        std_score           : float — std dev of similarities
        score_spread        : float — max – min similarity
        label_match_rate    : float | None — fraction matching query_label
        quality_tier        : str   — "strong" | "moderate" | "weak"
    """
    if not retrieved:
        return _empty_report()

    scores = np.array([r["similarity_score"] for r in retrieved])

    top1_score  = float(scores[0])
    mean_score  = float(scores.mean())
    std_score   = float(scores.std()) if len(scores) > 1 else 0.0
    score_spread = float(scores.max() - scores.min()) if len(scores) > 1 else 0.0
    n_above     = int((scores >= relevant_threshold).sum())

    # ------------------------------------------------------------------
    # Label-match rate (only computable if the query label is known)
    # ------------------------------------------------------------------
    label_match_rate = None
    if query_label is not None:
        matches = sum(
            1 for r in retrieved
            if r.get("primary_label", "").lower() == query_label.lower()
        )
        label_match_rate = matches / len(retrieved)

    # ------------------------------------------------------------------
    # Quality tier classification
    # ------------------------------------------------------------------
    quality_tier = _classify_quality(top1_score, mean_score)

    return {
        "n_retrieved"       : len(retrieved),
        "n_above_threshold" : n_above,
        "top1_score"        : top1_score,
        "mean_score"        : mean_score,
        "std_score"         : std_score,
        "score_spread"      : score_spread,
        "label_match_rate"  : label_match_rate,
        "quality_tier"      : quality_tier,
    }


def _classify_quality(top1: float, mean: float) -> str:
    """
    Map numerical scores to a human-readable quality tier.

    Thresholds (calibrated for all-MiniLM-L6-v2 on ticket data):
      strong   : top1 ≥ 0.75 and mean ≥ 0.60
      moderate : top1 ≥ 0.55 or  mean ≥ 0.50
      weak     : otherwise
    """
    if top1 >= 0.75 and mean >= 0.60:
        return "strong"
    if top1 >= 0.55 or mean >= 0.50:
        return "moderate"
    return "weak"


def _empty_report() -> dict:
    return {
        "n_retrieved"       : 0,
        "n_above_threshold" : 0,
        "top1_score"        : 0.0,
        "mean_score"        : 0.0,
        "std_score"         : 0.0,
        "score_spread"      : 0.0,
        "label_match_rate"  : None,
        "quality_tier"      : "weak",
    }


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import pprint

    # Simulate a retrieval result
    fake_results = [
        {"rank": 1, "issue_number": 1001, "primary_label": "billing",
         "similarity_score": 0.88, "solution_comments": "…", "clean_text": "…"},
        {"rank": 2, "issue_number": 1002, "primary_label": "billing",
         "similarity_score": 0.82, "solution_comments": "…", "clean_text": "…"},
        {"rank": 3, "issue_number": 1003, "primary_label": "duplicate",
         "similarity_score": 0.74, "solution_comments": "…", "clean_text": "…"},
    ]

    report = evaluate_retrieval(fake_results, query_label="billing")
    print("=== Retrieval Evaluation Report ===")
    pprint.pprint(report)
