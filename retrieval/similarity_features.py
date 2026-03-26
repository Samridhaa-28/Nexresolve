
import numpy as np


# ---------------------------------------------------------------------------
# Core feature computation
# ---------------------------------------------------------------------------

def compute_similarity_features(retrieved: list[dict]) -> dict:
    
    if not retrieved:
        return {"max_sim": 0.0, "avg_sim": 0.0, "sim_spread": 0.0}

    scores = np.array([r["similarity_score"] for r in retrieved], dtype=float)

    max_sim    = float(scores.max())
    avg_sim    = float(scores.mean())
    sim_spread = float(scores.max() - scores.min()) if len(scores) > 1 else 0.0

    return {
        "max_sim"    : round(max_sim,    4),
        "avg_sim"    : round(avg_sim,    4),
        "sim_spread" : round(sim_spread, 4),
    }


# ---------------------------------------------------------------------------
# Interpretation utility (for logging / debugging)
# ---------------------------------------------------------------------------

def interpret_similarity_features(features: dict) -> str:
    """
    Human-readable interpretation of the similarity features.

    Returns a short diagnostic string useful for debugging.
    """
    max_sim    = features.get("max_sim",    0.0)
    avg_sim    = features.get("avg_sim",    0.0)
    sim_spread = features.get("sim_spread", 0.0)

    if max_sim >= 0.75 and avg_sim >= 0.60:
        quality = "STRONG — reliable retrieval evidence"
    elif max_sim >= 0.55:
        quality = "MODERATE — some retrieval evidence"
    else:
        quality = "WEAK — retrieval unreliable"

    spread_note = (
        "consistent matches" if sim_spread < 0.15
        else "varied match quality (only top-1 is strong)"
    )

    return (
        f"max_sim={max_sim:.3f}, avg_sim={avg_sim:.3f}, "
        f"sim_spread={sim_spread:.3f} → {quality} | {spread_note}"
    )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Strong retrieval scenario
    strong = [
        {"similarity_score": 0.91},
        {"similarity_score": 0.87},
        {"similarity_score": 0.82},
    ]
    feats_strong = compute_similarity_features(strong)
    print("STRONG scenario:", feats_strong)
    print(" ", interpret_similarity_features(feats_strong))
    print()

    # Weak retrieval scenario
    weak = [
        {"similarity_score": 0.39},
        {"similarity_score": 0.34},
        {"similarity_score": 0.30},
    ]
    feats_weak = compute_similarity_features(weak)
    print("WEAK scenario:", feats_weak)
    print(" ", interpret_similarity_features(feats_weak))
