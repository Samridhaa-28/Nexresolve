import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Optional
from rl.action_space import Strategy, get_action_strategy, get_strategy_actions
from rl.training_utils import epsilon_scheduler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQNNetwork(nn.Module):
    """
    Feed-forward neural network for Q-value estimation.
    Architecture: 37 -> 256 -> 256 -> 4
    """
    def __init__(self, input_dim: int = 37, output_dim: int = 4):
        super(DQNNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

class Level1Agent:
    """
    Agent responsible for selecting high-level strategies (ROUTE, CLARIFY, SUGGEST, ESCALATE).
    """
    def __init__(self, state_dim: int = 37, gamma: float = 0.95, epsilon_start: float = 1.0, 
                 epsilon_min: float = 0.05, epsilon_decay_steps: int = 5000):
        self.state_dim = state_dim
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay_steps = epsilon_decay_steps
        self.steps_done = 0
        
        self.online_net = DQNNetwork(state_dim, 4).to(device)
        self.target_net = DQNNetwork(state_dim, 4).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=3e-4)
        self.loss_fn = nn.MSELoss()
        
        self.strategy_counts = {s.name: 0 for s in Strategy}

    def _get_strategy_mask(self, action_mask: np.ndarray) -> np.ndarray:
        """
        Derives strategy validity from a 17-dimensional action mask.
        A strategy is valid if at least one of its actions is valid.
        """
        strat_mask = np.zeros(4, dtype=np.float32)
        for strat in Strategy:
            indices = get_strategy_actions(strat)
            if any(action_mask[idx] == 1.0 for idx in indices):
                strat_mask[strat.value] = 1.0
        return strat_mask

    def select_action(self, state: np.ndarray, action_mask: np.ndarray, greedy=False) -> int:
        """
        Selects a strategy using epsilon-greedy policy with masking.
        """
        if greedy:
            epsilon = 0.0
        else:
            epsilon = epsilon_scheduler(self.steps_done, self.epsilon_start, self.epsilon_min, self.epsilon_decay_steps)
        self.steps_done += 1
        
        strat_mask = self._get_strategy_mask(action_mask)
        
        if np.random.random() < epsilon:
            # Random valid strategy
            valid_strats = [i for i, m in enumerate(strat_mask) if m == 1.0]
            strategy_idx = np.random.choice(valid_strats)
        else:
            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = self.online_net(state_t).cpu().numpy()[0]
            
            # Apply mask: invalid strategies get Q = -1e9
            masked_q = np.where(strat_mask == 1.0, q_values, -1e9)
            strategy_idx = np.argmax(masked_q)
            
        self.strategy_counts[Strategy(strategy_idx).name] += 1
        return strategy_idx

    def update(self, batch) -> float:
        """
        Performs one gradient descent step on a batch of transitions.
        Expects batch = (states, actions, rewards, next_states, dones)
        where actions are global action IDs.
        """
        states, global_actions, rewards, next_states, dones = batch
        
        # Convert global action IDs to strategy indices
        # 0-5 -> 0, 6-11 -> 1, 12-15 -> 2, 16 -> 3
        strategy_indices = []
        for a in global_actions:
            if 0 <= a <= 5: strategy_indices.append(0)
            elif 6 <= a <= 11: strategy_indices.append(1)
            elif 12 <= a <= 15: strategy_indices.append(2)
            elif a == 16: strategy_indices.append(3)
        
        # Convert to tensors
        states_t = torch.FloatTensor(states).to(device)
        strategy_indices_t = torch.LongTensor(strategy_indices).to(device)
        rewards_t = torch.FloatTensor(rewards).to(device)
        next_states_t = torch.FloatTensor(next_states).to(device)
        dones_t = torch.FloatTensor(dones.astype(np.float32)).to(device)
        
        # Current Q-values
        q_values = self.online_net(states_t)
        q_pred = q_values.gather(1, strategy_indices_t.unsqueeze(1)).squeeze(1)
        
        # Next Q-values from target network
        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(1)[0].detach()
            target = rewards_t + self.gamma * next_q * (1 - dones_t)
            
        loss = self.loss_fn(q_pred, target)
        
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'online_state_dict': self.online_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps_done': self.steps_done
        }, path)

    def load(self, path: str):
        if not os.path.exists(path):
            return
        checkpoint = torch.load(path, map_location=device)
        self.online_net.load_state_dict(checkpoint['online_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.steps_done = checkpoint.get('steps_done', 0)
