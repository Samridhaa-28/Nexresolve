import numpy as np
from typing import Dict, Any

from evaluation.baseline_agents import RandomAgent, RuleBasedAgent
from rl.action_space import get_action_strategy, Strategy

# Re-export existing agents for convenience
__all__ = ["RandomAgent", "RuleBasedAgent", "RetrievalOnlyAgent"]


class RetrievalOnlyAgent:
    """
    Baseline agent that uses retrieval similarity only (no learning).

    Decision rule:
        max_similarity >= 0.50  ->  SUGGEST (first valid suggest action)
        max_similarity <  0.50  ->  ROUTE   (first valid route action)

    Always respects action masking. Falls back to any valid action if the
    preferred strategy is fully masked.
    """

    def select_action(self, state: np.ndarray, mask: np.ndarray, info: Dict[str, Any]) -> int:
        valid_actions = np.where(mask == 1)[0]
        if len(valid_actions) == 0:
            # Safety fallback — environment guarantees at least one valid action,
            # but guard defensively.
            return 0

        # Extract similarity score using all known key variants
        max_sim = info.get("max_similarity", info.get("max_sim", info.get("similarity", 0.0)))

        if max_sim >= 0.50:
            suggest_actions = [a for a in valid_actions if get_action_strategy(a) == Strategy.SUGGEST]
            if suggest_actions:
                return int(suggest_actions[0])

        # Default: ROUTE (or fallback if ROUTE is also masked)
        route_actions = [a for a in valid_actions if get_action_strategy(a) == Strategy.ROUTE]
        if route_actions:
            return int(route_actions[0])

        # Final fallback: any valid action
        return int(valid_actions[0])
