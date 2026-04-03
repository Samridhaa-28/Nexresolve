import numpy as np
import logging

def epsilon_scheduler(step: int, start: float = 1.0, end: float = 0.05, steps: int = 5000) -> float:
    """
    Linearly decays epsilon for epsilon-greedy exploration.
    """
    if step >= steps:
        return end
    return start - (start - end) * (step / steps)

def bandit_mix_scheduler(step: int) -> float:
    """
    Decays the mixing ratio of bandit actions into the training loop.
    0.7 -> 0.2 over 2000 steps, then drops to 0.
    """
    if step < 2000:
        return 0.7 - (0.7 - 0.2) * (step / 2000)
    return 0.0

class TrainingLogger:
    """
    Helper for tracking and summarizing RL metrics.
    """
    def __init__(self):
        self.rewards = []
        self.q_values = []
        self.strategy_dist = {}

    def log_step(self, reward, q_val, strategy_name):
        self.rewards.append(reward)
        self.q_values.append(q_val)
        self.strategy_dist[strategy_name] = self.strategy_dist.get(strategy_name, 0) + 1

    def get_summary(self, last_n: int = 100):
        avg_reward = np.mean(self.rewards[-last_n:]) if self.rewards else 0.0
        avg_q = np.mean(self.q_values[-last_n:]) if self.q_values else 0.0
        
        return {
            "avg_reward": avg_reward,
            "avg_q_value": avg_q,
            "strategy_distribution": self.strategy_dist
        }
    
    def clear_dist(self):
        self.strategy_dist = {}
