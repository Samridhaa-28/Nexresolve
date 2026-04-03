import os
import sys
import torch
import numpy as np
import pandas as pd
import pickle
import logging
from datetime import datetime

# Ensure root dir is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rl.environment import NexResolveEnv
from rl.level1_dqn import Level1Agent
from rl.level2_dqn import Level2Agent
from rl.replay_buffer import ReplayBuffer
from rl.action_masking import get_action_mask
from rl.action_space import Strategy

# Constants
GAMMA = 0.95
LR = 3e-4
BATCH_SIZE = 64
EPISODES = 1500  # 🔥 UPDATED
MAX_STEPS = 6
TARGET_UPDATE = 200
LEARNING_STARTS = 500  # 🔥 UPDATED

# Paths
SCALER_PATH = "models/rl/state_scaler.pkl"
BUFFER_PATH = "models/rl/replay_buffer_normalized.pkl"
MODEL_SAVE_DIR = "models/rl/"
LOG_DIR = "evaluation/reports/"


def train():
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Setup Logging
    log_file = os.path.join(LOG_DIR, f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ])
    logger = logging.getLogger()
    logger.info("Starting RL Training for NexResolve")

    # Load Scaler
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    logger.info("State scaler loaded")

    # Initialize Environment
    env = NexResolveEnv()
    
    # Initialize Agents
    l1_agent = Level1Agent(gamma=GAMMA)
    l2_agent = Level2Agent(gamma=GAMMA)
    logger.info("L1 and L2 Agents initialized")

    # Initialize and Preload Buffer
    buffer = ReplayBuffer(capacity=50000)
    if os.path.exists(BUFFER_PATH):
        with open(BUFFER_PATH, 'rb') as f:
            warm_start_data = pickle.load(f)
            for transition in warm_start_data:
                buffer.add(*transition)
        logger.info(f"Preloaded {len(warm_start_data)} transitions into buffer")
    else:
        logger.warning("No warm-start buffer found. Training from scratch.")

    # Training Stats
    stats = {
        "rewards": [],
        "success_rate": [],
        "steps": [],
        "sla_breaches": 0,
        "strategy_dist": {s.name: 0 for s in Strategy}
    }

    # Loss tracking
    l1_loss_history = []
    l2_loss_history = []

    # 🔥 NEW: Best model tracking
    best_avg_reward = -float("inf")
    
    total_steps = 0
    
    for episode in range(1, EPISODES + 1):
        state = env.reset()
        state = scaler.transform([state])[0]
        
        episode_reward = 0
        done = False
        steps = 0
        
        while not done and steps < MAX_STEPS:
            action_mask = get_action_mask(state)
            
            # Select Strategy (L1)
            strategy = l1_agent.select_action(state, action_mask)
            
            # Select Action (L2)
            action = l2_agent.select_action(state, strategy, action_mask)
            
            # Step Env
            next_state_raw, reward, done, info = env.step(action)
            next_state = scaler.transform([next_state_raw])[0]
            
            # Store transition
            buffer.add(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            steps += 1
            total_steps += 1
            
            # Update Strategy Stats
            stats["strategy_dist"][Strategy(strategy).name] += 1
            
            # Perform Updates
            if len(buffer) > LEARNING_STARTS:
                batch = buffer.sample(BATCH_SIZE)
                
                l1_loss = l1_agent.update(batch)
                l2_losses = l2_agent.update(batch)
                
                if l1_loss is not None:
                    l1_loss_history.append(l1_loss)
                
                if l2_losses:
                    l2_loss_history.extend(list(l2_losses.values()))
            
            # Target Update
            if total_steps % TARGET_UPDATE == 0:
                l1_agent.update_target_network()
                l2_agent.update_target_networks()
                
        # Post-episode tracking
        stats["rewards"].append(episode_reward)
        stats["steps"].append(steps)
        
        if info.get("success", False):
            stats["success_rate"].append(1)
        else:
            stats["success_rate"].append(0)
             
        if steps >= MAX_STEPS or (isinstance(state, np.ndarray) and state[5] == 1.0):
            stats["sla_breaches"] += 1

        # Periodic Logging
        if episode % 50 == 0:
            avg_reward = np.mean(stats["rewards"][-50:])
            avg_steps = np.mean(stats["steps"][-50:])
            success_pct = np.mean(stats["success_rate"][-50:]) * 100

            avg_l1_loss = np.mean(l1_loss_history[-100:]) if l1_loss_history else 0
            avg_l2_loss = np.mean(l2_loss_history[-100:]) if l2_loss_history else 0
            
            total_strat_calls = sum(stats["strategy_dist"].values())
            dist_str = ""
            for name, count in stats["strategy_dist"].items():
                pct = (count / total_strat_calls) * 100
                dist_str += f"{name}: {pct:.1f}% "
                if pct > 90:
                    logger.warning(f"WARNING: Strategy {name} collapse detected ({pct:.1f}%)")
            
            logger.info(
                f"Ep {episode:4d} | Reward: {avg_reward:6.2f} | Steps: {avg_steps:.1f} | "
                f"Success: {success_pct:5.1f}% | L1 Loss: {avg_l1_loss:.3f} | L2 Loss: {avg_l2_loss:.3f} | {dist_str}"
            )

            # 🔥 SAVE BEST MODEL
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                l1_agent.save(os.path.join(MODEL_SAVE_DIR, "l1_best.pth"))
                l2_agent.save_all(os.path.join(MODEL_SAVE_DIR, "best"))
                logger.info(f"New best model saved (Reward: {avg_reward:.2f})")

            stats["strategy_dist"] = {s.name: 0 for s in Strategy}

    # Save Final Models
    l1_agent.save(os.path.join(MODEL_SAVE_DIR, "l1_final.pth"))
    l2_agent.save_all(MODEL_SAVE_DIR)
    logger.info(f"Training complete. Models saved to {MODEL_SAVE_DIR}")


if __name__ == "__main__":
    train()