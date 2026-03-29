import numpy as np
from rl.action_space import is_clarify, is_suggest, is_escalate


def compute_reward(prev_state: np.ndarray, action_id: int, next_state: np.ndarray) -> float:

    reward = 0.0

    # =========================================================
    # BASE REWARDS
    # =========================================================

    if next_state[30] == 1.0 and prev_state[30] == 0.0:
        reward += 25.0

        if next_state[5] == 0.0:
            reward += 5.0

        if prev_state[4] <= 2:
            reward += 5.0

    if next_state[30] == 0.0:
        reward -= 0.1

    if next_state[22] > prev_state[22]:
        reward -= 2.0

    if next_state[5] == 1.0 and prev_state[5] == 0.0:
        reward -= 10.0

    delta_f = next_state[27] - prev_state[27]
    if delta_f > 0:
        reward -= 2.0 * delta_f

    # =========================================================
    # STRATEGY SIGNALS
    # =========================================================

    if is_clarify(action_id) and prev_state[23] == 0.0:
        reward -= 1.5

    if is_suggest(action_id):

        if next_state[30] == 1.0:
            reward += 3.0

            if next_state[35] == 1:
                reward += 2.0

            if next_state[36] == 1:
                reward -= 0.5
        else:
            reward -= 1.5

        if prev_state[31] < 0.4:
            reward -= 1.5

    if is_escalate(action_id):
        if prev_state[27] < 0.5:
            reward -= 5.0

    return float(np.clip(reward, -15, 40))