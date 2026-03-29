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

    def reset(self, ticket_idx: Optional[int] = None) -> np.ndarray:

        if ticket_idx is None:
            self.current_ticket_idx = np.random.randint(0, len(self.df))
        else:
            self.current_ticket_idx = ticket_idx % len(self.df)

        row = self.df.iloc[self.current_ticket_idx]

        from rl.state_builder import STATE_COLUMNS
        self.state = row[STATE_COLUMNS].values.astype(np.float32)

        self.done = False
        return self.state.copy()

    def step(self, action_id: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:

        if self.done:
            raise RuntimeError("Episode finished. Call reset().")

        prev_state = self.state.copy()
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
        next_state[4] += 1.0
        next_state[8] += 1.0

        next_state[6] = max(0.0, next_state[6] - 0.05)

        if next_state[6] <= 0.0:
            next_state[5] = 1.0
            self.done = True

        if next_state[4] >= 6:
            self.done = True

        self.state = next_state

        reward = compute_reward(prev_state, action_id, next_state)

        info = {
            "ticket_id": self.current_ticket_idx,
            "action": action_name,
            "reward": reward
        }

        return self.state.copy(), reward, self.done, info

    # =========================================================
    # ACTION LOGIC
    # =========================================================

    def _apply_clarify(self, state: np.ndarray, action_id: int) -> np.ndarray:

        action_name = INDEX_TO_ACTION[action_id]

        if action_name == "ask_error_type":
            state[16] = 1.0
        elif action_name == "ask_version":
            state[15] = 1.0
        elif action_name == "ask_platform":
            state[17] = 1.0
        elif action_name == "ask_hardware":
            state[18] = 1.0

        state[20] = max(0.0, state[20] - 1.0)
        state[21] = min(1.0, state[21] + 0.1)

        state[27] = min(1.0, state[27] + 0.05)
        state[25] = np.clip(state[25] - 0.02, -1.0, 1.0)

        state[31] = min(1.0, state[31] + 0.15)

        if state[21] > 0.8:
            state[34] = 0.0

        if state[15] and state[16] and state[17] and state[18]:
            state[23] = 0.0

        return state

    def _apply_route(self, state: np.ndarray, action_id: int) -> np.ndarray:

        state[22] += 1.0
        state[6] = max(0.0, state[6] - 0.03)
        state[31] = min(1.0, state[31] + 0.05)

        return state

    def _apply_suggest(self, state: np.ndarray, action_id: int) -> np.ndarray:

        max_sim = state[31]

        if max_sim >= 0.55:
            state[30] = 1.0
            self.done = True
        else:
            state[27] = min(1.0, state[27] + 0.15)
            state[29] = min(5.0, state[29] + 1.0)

        return state

    def _apply_escalate(self, state: np.ndarray, action_id: int) -> np.ndarray:
        self.done = True
        return state