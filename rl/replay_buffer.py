import random
from collections import deque
import numpy as np
import torch

class ReplayBuffer:
    """
    Circular buffer for storing and sampling reinforcement learning transitions.
    """
    def __init__(self, capacity: int = 50000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        """
        Adds a transition to the buffer.
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        """
        Samples a random batch of transitions from the buffer.
        Returns them as a tuple of numpy arrays/lists.
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.bool_)
        )

    def __len__(self):
        return len(self.buffer)
