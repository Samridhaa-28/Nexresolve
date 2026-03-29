import numpy as np
import pickle
from typing import List, Optional, Tuple
from rl.action_space import get_action_count
from rl.action_masking import get_action_mask
from rl.environment import NexResolveEnv


# =========================================================
# 🔹 LINUCB
# =========================================================
class LinUCB:
    def __init__(self, context_dim: int = 37, alpha: float = 1.0):
        self.d = context_dim
        self.alpha = alpha
        self.n_actions = get_action_count()

        self.A = [np.identity(self.d) for _ in range(self.n_actions)]
        self.b = [np.zeros((self.d, 1)) for _ in range(self.n_actions)]

    def select_action(self, context: np.ndarray, mask: Optional[np.ndarray] = None) -> int:
        context = context.reshape(-1, 1)
        p = np.zeros(self.n_actions)

        for a in range(self.n_actions):
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]
            p[a] = theta.T @ context + self.alpha * np.sqrt(context.T @ A_inv @ context)

        if mask is not None:
            p = np.where(mask == 1.0, p, -np.inf)

        if np.all(np.isneginf(p)):
            return np.random.randint(self.n_actions)

        return int(np.argmax(p))

    def update(self, context: np.ndarray, action: int, reward: float):
        context = context.reshape(-1, 1)
        self.A[action] += context @ context.T
        self.b[action] += reward * context

    def reset(self):
        self.A = [np.identity(self.d) for _ in range(self.n_actions)]
        self.b = [np.zeros((self.d, 1)) for _ in range(self.n_actions)]


# =========================================================
# 🔹 THOMPSON SAMPLING
# =========================================================
class ThompsonSampling:
    def __init__(self, context_dim: int = 37, v: float = 0.1):
        self.d = context_dim
        self.v = v
        self.n_actions = get_action_count()

        self.B = [np.identity(self.d) for _ in range(self.n_actions)]
        self.f = [np.zeros(self.d) for _ in range(self.n_actions)]
        self.mu = [np.zeros(self.d) for _ in range(self.n_actions)]

    def select_action(self, context: np.ndarray, mask: Optional[np.ndarray] = None) -> int:
        p = np.zeros(self.n_actions)

        for a in range(self.n_actions):
            B_inv = np.linalg.inv(self.B[a])
            theta_sampled = np.random.multivariate_normal(self.mu[a], self.v**2 * B_inv)
            p[a] = np.dot(theta_sampled, context)

        if mask is not None:
            p = np.where(mask == 1.0, p, -np.inf)

        if np.all(np.isneginf(p)):
            return np.random.randint(self.n_actions)

        return int(np.argmax(p))

    def update(self, context: np.ndarray, action: int, reward: float):
        self.B[action] += np.outer(context, context)
        self.f[action] += reward * context
        self.mu[action] = np.linalg.inv(self.B[action]) @ self.f[action]

    def reset(self):
        self.B = [np.identity(self.d) for _ in range(self.n_actions)]
        self.f = [np.zeros(self.d) for _ in range(self.n_actions)]
        self.mu = [np.zeros(self.d) for _ in range(self.n_actions)]


# =========================================================
# 🔥 REPLAY BUFFER GENERATION (FOR DQN)
# =========================================================
def generate_replay_buffer(
    bandit,
    num_episodes: int = 500,
    max_steps: int = 6,
    buffer_size: int = 10000,
    save_path: str = "models/rl/replay_buffer.pkl"
):
    """
    Runs bandit policy and collects experience for DQN.
    """

    env = NexResolveEnv()
    replay_buffer = []

    for ep in range(num_episodes):
        state = env.reset()
        done = False
        step = 0

        while not done and step < max_steps:

            mask = get_action_mask(state)
            action = bandit.select_action(state, mask)

            next_state, reward, done, info = env.step(action)

            # 🔥 STORE EXPERIENCE
            replay_buffer.append((state, action, reward, next_state, done))

            # 🔥 LIMIT BUFFER SIZE
            if len(replay_buffer) > buffer_size:
                replay_buffer.pop(0)

            bandit.update(state, action, reward)

            state = next_state
            step += 1

    # 🔥 SAVE BUFFER
    with open(save_path, "wb") as f:
        pickle.dump(replay_buffer, f)

    print(f"\nReplay buffer saved: {save_path}")
    print(f"Total samples: {len(replay_buffer)}")

    return replay_buffer


# =========================================================
# 🔹 MAIN TEST
# =========================================================
if __name__ == "__main__":

    # 🔹 Choose bandit
    bandit = LinUCB()  # or ThompsonSampling()

    # 🔹 Generate replay buffer
    replay_buffer = generate_replay_buffer(
        bandit=bandit,
        num_episodes=100,
        max_steps=10
    )

    print("\nSample experience:")
    print(replay_buffer[0])