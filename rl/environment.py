import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional

from rl.action_space import (
    INDEX_TO_ACTION, Strategy, get_action_strategy
)
from rl.reward import compute_reward


class NexResolveEnv:

    def __init__(self, dataset_path: str = "data/final/rl_ready_dataset.csv"):
        self.df = pd.read_csv(dataset_path)
        self.state = None
        self.current_ticket_idx = -1
        self.done = False

        # --- INTERNAL METRICS ---
        self.sla_breach = False
        self.frustration_level = 0.0
        self.resolution_success = False
        self.last_info = {}

        # --- COLUMN MAPPING (Avoid magic numbers) ---
        from rl.state_builder import STATE_COLUMNS
        self.COL_IDX = {col: i for i, col in enumerate(STATE_COLUMNS)}

    def reset(self, ticket_idx: Optional[int] = None) -> np.ndarray:

        if ticket_idx is None:
            self.current_ticket_idx = np.random.randint(0, len(self.df))
        else:
            self.current_ticket_idx = ticket_idx % len(self.df)

        row = self.df.iloc[self.current_ticket_idx]

        from rl.state_builder import STATE_COLUMNS
        self.state = row[STATE_COLUMNS].values.astype(np.float32)

        # Sync internal metrics
        self.sla_breach = bool(self.state[self.COL_IDX["sla_breach_flag"]])
        self.frustration_level = float(self.state[self.COL_IDX["frustration_level"]])
        self.resolution_success = bool(self.state[self.COL_IDX["resolution_success"]])

        self.done = False
        self.last_info = {
            "success": self.resolution_success,
            "sla_breach": self.sla_breach,
            "frustration": self.frustration_level,
            "frustration_delta": 0.0,
            "max_sim": float(self.state[self.COL_IDX["max_sim"]])
        }
        return self.state.copy()

    def get_current_info(self) -> Dict[str, Any]:
        """Return the most recent info dict (useful for rule-based agents)."""
        return self.last_info

    def step(self, action_id: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:

        if self.done:
            raise RuntimeError("Episode finished. Call reset().")

        prev_state = self.state.copy()
        prev_frustration = self.frustration_level

        action_name = INDEX_TO_ACTION[action_id]
        strategy = get_action_strategy(action_id)

        next_state = self.state.copy()

        # APPLY ACTION
        if strategy == Strategy.CLARIFY:
            next_state = self._apply_clarify(next_state, action_id)

        elif strategy == Strategy.ROUTE:
            next_state = self._apply_route(next_state, action_id)

        elif strategy == Strategy.SUGGEST:
            next_state = self._apply_suggest(next_state, action_id)

        elif strategy == Strategy.ESCALATE:
            next_state = self._apply_escalate(next_state, action_id)

        # GLOBAL UPDATES
        next_state[self.COL_IDX["turn_count"]] += 1.0
        next_state[self.COL_IDX["interaction_depth"]] += 1.0

        # Decrement SLA
        sla_val = next_state[self.COL_IDX["sla_remaining_norm"]]
        sla_val = max(0.0, sla_val - 0.05)
        next_state[self.COL_IDX["sla_remaining_norm"]] = sla_val

        if sla_val <= 0.0:
            next_state[self.COL_IDX["sla_breach_flag"]] = 1.0
            self.sla_breach = True
            self.done = True

        if next_state[self.COL_IDX["turn_count"]] >= 6:
            self.done = True

        self.state = next_state
        self.frustration_level = float(self.state[self.COL_IDX["frustration_level"]])
        self.resolution_success = bool(self.state[self.COL_IDX["resolution_success"]])

        reward = compute_reward(prev_state, action_id, next_state)

        success = self.resolution_success
        if success:
            self.done = True

        info = {
            "ticket_id": self.current_ticket_idx,
            "action": action_name,
            "reward": reward,
            "success": success,
            "sla_breach": self.sla_breach,
            "frustration": self.frustration_level,
            "frustration_delta": self.frustration_level - prev_frustration,
            "max_sim": float(self.state[self.COL_IDX["max_sim"]])
        }
        self.last_info = info

        return self.state.copy(), reward, self.done, info

    # =========================================================
    # ACTION LOGIC
    # =========================================================

    def _apply_clarify(self, state: np.ndarray, action_id: int) -> np.ndarray:

        action_name = INDEX_TO_ACTION[action_id]

        if action_name == "ask_error_type":
            state[self.COL_IDX["has_error_type"]] = 1.0
        elif action_name == "ask_version":
            state[self.COL_IDX["has_version"]] = 1.0
        elif action_name == "ask_platform":
            state[self.COL_IDX["has_platform"]] = 1.0
        elif action_name == "ask_hardware":
            state[self.COL_IDX["has_hardware"]] = 1.0

        # Updating missing count and completeness
        state[self.COL_IDX["missing_count"]] = max(0.0, state[self.COL_IDX["missing_count"]] - 1.0)
        state[self.COL_IDX["completeness_score"]] = min(1.0, state[self.COL_IDX["completeness_score"]] + 0.1)

        # Frustration and sentiment side effects
        state[self.COL_IDX["frustration_level"]] = min(1.0, state[self.COL_IDX["frustration_level"]] + 0.05)
        state[self.COL_IDX["sentiment_score"]] = np.clip(state[self.COL_IDX["sentiment_score"]] - 0.02, -1.0, 1.0)

        # Sim side effect (clarification improves understanding/similarity)
        state[self.COL_IDX["max_sim"]] = min(1.0, state[self.COL_IDX["max_sim"]] + 0.15)

        # If completeness is high, reset needs_clarification
        if state[self.COL_IDX["completeness_score"]] > 0.8:
            state[self.COL_IDX["knowledge_gap_flag"]] = 0.0 # Just as a logic example

        if (state[self.COL_IDX["has_version"]] and state[self.COL_IDX["has_error_type"]] and 
            state[self.COL_IDX["has_platform"]] and state[self.COL_IDX["has_hardware"]]):
            state[self.COL_IDX["needs_clarification"]] = 0.0

        return state

    def _apply_route(self, state: np.ndarray, action_id: int) -> np.ndarray:

        state[self.COL_IDX["reassignment_count"]] += 1.0
        # More reassignment reduces SLA faster
        state[self.COL_IDX["sla_remaining_norm"]] = max(0.0, state[self.COL_IDX["sla_remaining_norm"]] - 0.03)
        state[self.COL_IDX["max_sim"]] = min(1.0, state[self.COL_IDX["max_sim"]] + 0.05)

        return state

    def _apply_suggest(self, state: np.ndarray, action_id: int) -> np.ndarray:

        max_sim = state[self.COL_IDX["max_sim"]]

        # Threshold for success
        if max_sim >= 0.55:
            state[self.COL_IDX["resolution_success"]] = 1.0
            self.done = True
        else:
            # Increase frustration and turn count penalty
            state[self.COL_IDX["frustration_level"]] = min(1.0, state[self.COL_IDX["frustration_level"]] + 0.15)
            state[self.COL_IDX["reopen_count"]] = min(5.0, state[self.COL_IDX["reopen_count"]] + 1.0)

        return state

    def _apply_escalate(self, state: np.ndarray, action_id: int) -> np.ndarray:
        self.done = True
        return state