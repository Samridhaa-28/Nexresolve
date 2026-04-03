import numpy as np
import random
from typing import Optional, Dict, Any
from rl.action_space import get_action_count, get_action_strategy, Strategy
from rl.bandit_policy import LinUCB

class BaseEvalAgent:
    def select_action(self, state: np.ndarray, mask: np.ndarray, info: Dict[str, Any]) -> int:
        raise NotImplementedError

class RandomAgent(BaseEvalAgent):
    """Selects a random valid action from the mask."""
    def select_action(self, state: np.ndarray, mask: np.ndarray, info: Dict[str, Any]) -> int:
        valid_actions = np.where(mask == 1.0)[0]
        if len(valid_actions) == 0:
            return random.randint(0, get_action_count() - 1)
        return int(random.choice(valid_actions))

class RuleBasedAgent(BaseEvalAgent):
    """
    Implements a simple heuristic based on the info dict:
    1. If frustration is very high (> 0.8) -> ESCALATE (if valid)
    2. If similarity is high (> 0.5) -> SUGGEST (if valid)
    3. If missing info -> CLARIFY (if valid)
    4. Else -> ROUTE
    """
    def select_action(self, state: np.ndarray, mask: np.ndarray, info: Dict[str, Any]) -> int:
        valid_actions = np.where(mask == 1.0)[0]
        if len(valid_actions) == 0:
            return 0  # Default to first action if all masked (shouldn't happen)

        frustration = info.get("frustration", 0.0)
        max_sim = info.get("max_sim", 0.0)
        
        # 1. ESCALATE if frustrated
        escalate_actions = [a for a in valid_actions if get_action_strategy(a) == Strategy.ESCALATE]
        if frustration > 0.8 and escalate_actions:
            return int(escalate_actions[0])

        # 2. SUGGEST if we have a good match
        suggest_actions = [a for a in valid_actions if get_action_strategy(a) == Strategy.SUGGEST]
        if max_sim > 0.55 and suggest_actions:
            return int(random.choice(suggest_actions))

        # 3. CLARIFY if missing entities (Heuristic: completeness_score is in state, but let's use a simpler rule)
        clarify_actions = [a for a in valid_actions if get_action_strategy(a) == Strategy.CLARIFY]
        if max_sim < 0.4 and clarify_actions:
            return int(random.choice(clarify_actions))

        # 4. ROUTE as fallback
        route_actions = [a for a in valid_actions if get_action_strategy(a) == Strategy.ROUTE]
        if route_actions:
            return int(random.choice(route_actions))

        return int(random.choice(valid_actions))

class BanditAgent(BaseEvalAgent):
    """Wraps the LinUCB bandit policy for evaluation."""
    def __init__(self, context_dim: int = 37, alpha: float = 0.1):
        self.bandit = LinUCB(context_dim=context_dim, alpha=alpha)

    def select_action(self, state: np.ndarray, mask: np.ndarray, info: Dict[str, Any]) -> int:
        # NOTE: Bandit expects normalized/raw state vector as context
        return self.bandit.select_action(state, mask)

    def update(self, state: np.ndarray, action: int, reward: float):
        self.bandit.update(state, action, reward)
