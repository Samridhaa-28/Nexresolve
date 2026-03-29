import numpy as np
from typing import List
from rl.action_space import get_action_count, get_strategy_actions, Strategy, ACTIONS


def get_action_mask(state: np.ndarray) -> np.ndarray:

    mask = np.ones(get_action_count(), dtype=np.float32)

    # ── Extract features ──
    sla_remaining      = state[6]
    urgent_flag        = state[14]
    completeness_score = state[21]
    needs_clarify      = state[23]
    frustration        = state[27]
    max_sim            = state[31]
    tier1              = state[35]
    tier2              = state[36]

    # ── Strategy indices ──
    route_indices    = get_strategy_actions(Strategy.ROUTE)
    clarify_indices  = get_strategy_actions(Strategy.CLARIFY)
    suggest_indices  = get_strategy_actions(Strategy.SUGGEST)
    escalate_indices = get_strategy_actions(Strategy.ESCALATE)

    # =========================================================
    # CLARIFY LOGIC
    # =========================================================
    if needs_clarify == 0 and completeness_score > 0.95:
        mask[clarify_indices] = 0.0

    if frustration > 0.85:
        mask[clarify_indices] = 0.0

    if sla_remaining < 0.1:
        mask[clarify_indices] = 0.0

    if urgent_flag == 1:
        mask[clarify_indices] = 0.0

    # =========================================================
    # SUGGEST LOGIC (NO HARD GAP BLOCK)
    # =========================================================
    if max_sim < 0.4:
        mask[suggest_indices] = 0.0
    elif max_sim >= 0.4:
        mask[suggest_indices] = 1.0

    # Tier influence (soft)
    if tier1 == 1 and max_sim >= 0.5:
        mask[suggest_indices] = np.maximum(mask[suggest_indices], 1.0)

    # =========================================================
    # ESCALATE LOGIC
    # =========================================================
    if frustration < 0.3 and sla_remaining > 0.5:
        mask[escalate_indices] = 0.0

    # =========================================================
    # SAFETY FALLBACK
    # =========================================================
    if np.sum(mask) == 0:
        mask[escalate_indices] = 1.0

    return mask


def get_valid_actions(state: np.ndarray) -> List[int]:
    mask = get_action_mask(state)
    return [i for i, m in enumerate(mask) if m == 1.0]


def apply_mask_to_q_values(q_values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.where(mask == 1.0, q_values, -np.inf)