import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Optional
from rl.action_space import Strategy, get_strategy_actions, INDEX_TO_ACTION

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Level2Network(nn.Module):
    """
    Sub-network for action selection within a strategy.
    Architecture: 37 -> 128 -> 128 -> K (where K is number of actions in strategy)
    """
    def __init__(self, input_dim: int = 37, output_dim: int = 6):
        super(Level2Network, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

class Level2Agent:
    """
    Agent responsible for selecting specific actions within a chosen strategy.
    Contains separate networks for each strategy.
    """
    def __init__(self, state_dim: int = 37, gamma: float = 0.95):
        self.state_dim = state_dim
        self.gamma = gamma
        
        # Strategy output dimensions
        self.strats = {
            Strategy.ROUTE: 6,
            Strategy.CLARIFY: 6,
            Strategy.SUGGEST: 4,
            Strategy.ESCALATE: 1
        }
        
        self.online_nets = {}
        self.target_nets = {}
        self.optimizers = {}
        self.loss_fn = nn.MSELoss()
        
        for strat, out_dim in self.strats.items():
            self.online_nets[strat] = Level2Network(state_dim, out_dim).to(device)
            self.target_nets[strat] = Level2Network(state_dim, out_dim).to(device)
            self.target_nets[strat].load_state_dict(self.online_nets[strat].state_dict())
            self.target_nets[strat].eval()
            self.optimizers[strat] = optim.Adam(self.online_nets[strat].parameters(), lr=3e-4)

    def select_action(self, state: np.ndarray, strategy: int, action_mask: np.ndarray) -> int:
        """
        Selects an action within the given strategy using the associated sub-network.
        Returns the global action ID.
        """
        strat_enum = Strategy(strategy)
        global_indices = get_strategy_actions(strat_enum)
        
        # Map global mask to local mask
        local_mask = np.array([action_mask[idx] for idx in global_indices], dtype=np.float32)
        
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        online_net = self.online_nets[strat_enum]
        
        with torch.no_grad():
            q_values = online_net(state_t).cpu().numpy()[0]
            
        # Apply local mask: invalid actions get Q = -1e9
        # Special case for Escalate (only 1 action)
        if len(global_indices) == 1:
            local_idx = 0
        else:
            masked_q = np.where(local_mask == 1.0, q_values, -1e9)
            local_idx = np.argmax(masked_q)
            
        return global_indices[local_idx]

    def update(self, batch) -> Dict[int, float]:
        """
        Updates each strategy sub-network using only the relevant transitions from the batch.
        Expects batch = (states, actions, rewards, next_states, dones)
        where actions are global action IDs.
        """
        states, global_actions, rewards, next_states, dones = batch
        losses = {}

        # Strategy ranges
        # ROUTE: 0-5
        # CLARIFY: 6-11
        # SUGGEST: 12-15
        # ESCALATE: 16
        strategy_ranges = {
            Strategy.ROUTE: range(0, 6),
            Strategy.CLARIFY: range(6, 12),
            Strategy.SUGGEST: range(12, 16),
            Strategy.ESCALATE: range(16, 17)
        }

        for strat_enum, action_range in strategy_ranges.items():
            global_indices = get_strategy_actions(strat_enum)
            global_to_local = {idx: i for i, idx in enumerate(global_indices)}
            
            # Filter transitions belonging to this strategy
            relevant_indices = [i for i, action in enumerate(global_actions) if action in action_range]
            
            if len(relevant_indices) < 2:
                continue

            states_f = states[relevant_indices]
            actions_f = [global_to_local[global_actions[i]] for i in relevant_indices]
            rewards_f = rewards[relevant_indices]
            next_states_f = next_states[relevant_indices]
            dones_f = dones[relevant_indices]

            # Convert to tensors
            states_t = torch.FloatTensor(states_f).to(device)
            actions_t = torch.LongTensor(actions_f).unsqueeze(1).to(device)
            rewards_t = torch.FloatTensor(rewards_f).unsqueeze(1).to(device)
            next_states_t = torch.FloatTensor(next_states_f).to(device)
            dones_t = torch.FloatTensor(dones_f.astype(np.float32)).unsqueeze(1).to(device)
            
            online_net = self.online_nets[strat_enum]
            target_net = self.target_nets[strat_enum]
            optimizer = self.optimizers[strat_enum]
            
            # Current Q-values
            curr_q = online_net(states_t).gather(1, actions_t)
            
            # Next Q-values from target network
            with torch.no_grad():
                next_q = target_net(next_states_t).max(1)[0].unsqueeze(1).detach()
                target_q = rewards_t + (1 - dones_t) * self.gamma * next_q
                
            loss = self.loss_fn(curr_q, target_q)
            
            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(online_net.parameters(), 1.0)
            optimizer.step()
            
            losses[strat_enum.value] = loss.item()
        
        return losses

    def update_target_networks(self):
        for strat in self.strats:
            self.target_nets[strat].load_state_dict(self.online_nets[strat].state_dict())

    def save_all(self, base_path: str):
        os.makedirs(base_path, exist_ok=True)
        for strat in self.strats:
            path = os.path.join(base_path, f"l2_{strat.name.lower()}.pth")
            torch.save({
                'online_state_dict': self.online_nets[strat].state_dict(),
                'target_state_dict': self.target_nets[strat].state_dict(),
                'optimizer_state_dict': self.optimizers[strat].state_dict()
            }, path)

    def load_all(self, base_path: str):
        if not os.path.exists(base_path):
            return
        for strat in self.strats:
            path = os.path.join(base_path, f"l2_{strat.name.lower()}.pth")
            if os.path.exists(path):
                checkpoint = torch.load(path, map_location=device)
                self.online_nets[strat].load_state_dict(checkpoint['online_state_dict'])
                self.target_nets[strat].load_state_dict(checkpoint['target_state_dict'])
                self.optimizers[strat].load_state_dict(checkpoint['optimizer_state_dict'])
