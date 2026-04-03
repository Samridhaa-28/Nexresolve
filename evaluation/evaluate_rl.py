import os
import sys
import torch
import numpy as np
import pandas as pd
import json
import pickle
import random
from typing import Dict, List, Any

# Ensure root dir is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rl.environment import NexResolveEnv
from rl.level1_dqn import Level1Agent
from rl.level2_dqn import Level2Agent
from rl.action_masking import get_action_mask
from rl.action_space import INDEX_TO_ACTION, Strategy
from evaluation.baseline_agents import RandomAgent, RuleBasedAgent, BanditAgent

# Configuration
SEED = 42
NUM_EPISODES = 200
MAX_STEPS = 6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
SCALER_PATH = "models/rl/state_scaler.pkl"
L1_PATH = "models/rl/l1_best.pth"
L2_DIR = "models/rl/best"
REPORT_DIR = "evaluation/reports"

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def load_dqn_agent(state_dim=37):
    l1 = Level1Agent(state_dim=state_dim)
    l2 = Level2Agent(state_dim=state_dim)
    
    # Load weights
    if os.path.exists(L1_PATH):
        checkpoint = torch.load(L1_PATH, map_location=DEVICE)

        if "online_state_dict" in checkpoint:
            l1.online_net.load_state_dict(checkpoint["online_state_dict"])
        else:
            l1.online_net.load_state_dict(checkpoint)
        l1.online_net.eval()
    
    # Load L2 strategy nets
    strats = ["route", "clarify", "suggest", "escalate"]
    for s in strats:
        p = os.path.join(L2_DIR, f"l2_{s}.pth")
        if os.path.exists(p):
            checkpoint = torch.load(p, map_location=DEVICE)
            l2.online_nets[Strategy[s.upper()]].load_state_dict(checkpoint["online_state_dict"])
            l2.online_nets[Strategy[s.upper()]].eval()
            
    return l1, l2

def run_evaluation(env, agent_name, agent_obj, scaler, num_episodes=200):
    print(f"  Evaluating {agent_name} ...")
    
    episode_logs = []
    agg_metrics = {
        "rewards": [],
        "success_rate": [],
        "steps": [],
        "sla_breach_rate": [],
        "frustration_delta": [],
        "strategy_dist": {s.name: 0 for s in Strategy}
    }
    
    for ep in range(num_episodes):
        state_raw = env.reset(ticket_idx=ep) # Deterministic for evaluation
        state = scaler.transform([state_raw])[0]
        
        ep_reward = 0
        ep_steps = 0
        ep_done = False
        ep_history = []
        ep_frustration = 0
        
        while not ep_done and ep_steps < MAX_STEPS:
            mask = get_action_mask(state)
            info = env.get_current_info()
            
            # Action Selection
            if agent_name == "DQN":
                l1, l2 = agent_obj
                # DQN agents internally handle state scaling if passed raw? 
                # No, train_rl.py transforms first.
                strat_idx = l1.select_action(state, mask, greedy=True) # Greedy
                action = l2.select_action(state, strat_idx, mask)
                strat_name = Strategy(strat_idx).name
            else:
                action = agent_obj.select_action(state, mask, info)
                from rl.action_space import get_action_strategy
                strat_name = get_action_strategy(action).name
            
            # Step
            next_state_raw, reward, ep_done, next_info = env.step(action)
            ep_frustration += next_info["frustration_delta"]
            next_state = scaler.transform([next_state_raw])[0]
            
            # Log transition
            transition = {
                "step": ep_steps,
                "state": state.tolist(),
                "action": int(action),
                "action_name": INDEX_TO_ACTION[action],
                "strategy": strat_name,
                "reward": float(reward),
                "next_state": next_state.tolist(),
                "info": next_info
            }
            ep_history.append(transition)
            
            agg_metrics["strategy_dist"][strat_name] += 1
            
            state = next_state
            ep_reward += reward
            ep_steps += 1
        
        # End of episode
        final_info = env.get_current_info()
        agg_metrics["rewards"].append(ep_reward)
        agg_metrics["success_rate"].append(1 if final_info["success"] else 0)
        agg_metrics["steps"].append(ep_steps)
        agg_metrics["sla_breach_rate"].append(1 if final_info["sla_breach"] else 0)
        agg_metrics["frustration_delta"].append(ep_frustration)
        
        episode_logs.append({
            "episode": ep,
            "total_reward": ep_reward,
            "steps": ep_steps,
            "success": final_info["success"],
            "sla_breach": final_info["sla_breach"],
            "history": ep_history
        })
    total = sum(agg_metrics["strategy_dist"].values())
    strategy_dist = {k: (v / total if total > 0 else 0) for k, v in agg_metrics["strategy_dist"].items()}
        
    # Summary
    summary = {
        "agent": agent_name,
        "avg_reward": float(np.mean(agg_metrics["rewards"])),
        "success_rate": float(np.mean(agg_metrics["success_rate"])),
        "avg_steps": float(np.mean(agg_metrics["steps"])),
        "sla_breach_rate": float(np.mean(agg_metrics["sla_breach_rate"])),
        "avg_frustration_delta": float(np.mean(agg_metrics["frustration_delta"])),
        "strategy_distribution": strategy_dist
    }
    
    return summary, episode_logs

def main():
    seed_everything(SEED)
    os.makedirs(REPORT_DIR, exist_ok=True)
    
    # Load assets
    if not os.path.exists(SCALER_PATH):
        print(f"Error: Scaler not found at {SCALER_PATH}")
        return
        
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
        
    env = NexResolveEnv()
    state_dim = 37 # Consistent with STATE_COLUMNS
    
    # Initialize agents
    dqn_agents = load_dqn_agent(state_dim)
    random_agent = RandomAgent()
    rule_agent = RuleBasedAgent()
    bandit_agent = BanditAgent(context_dim=state_dim)
    
    # Warm-start Bandit on 100 episodes before evaluation
    print("Warm-starting Bandit agent ...")
    for i in range(100):
        s_raw = env.reset()
        s = scaler.transform([s_raw])[0]
        d = False
        while not d:
            m = get_action_mask(s)
            a = bandit_agent.select_action(s, m, {})
            ns_raw, r, d, _ = env.step(a)
            ns = scaler.transform([ns_raw])[0]
            bandit_agent.update(s, a, r)
            s = ns

    agents = {
        "DQN": dqn_agents,
        "Random": random_agent,
        "Rule-Based": rule_agent,
        "Bandit": bandit_agent
    }
    
    all_summaries = []
    all_episode_logs = {}
    
    print(f"Starting Evaluation on {NUM_EPISODES} episodes ...")
    for name, obj in agents.items():
        summary, logs = run_evaluation(env, name, obj, scaler, NUM_EPISODES)
        all_summaries.append(summary)
        all_episode_logs[name] = logs
        
    # Save Results
    with open(os.path.join(REPORT_DIR, "rl_metrics.json"), "w") as f:
        json.dump(all_summaries, f, indent=2)
        
    with open(os.path.join(REPORT_DIR, "rl_episode_logs.json"), "w") as f:
        json.dump(all_episode_logs, f)
        
    # Print Summary Table
    df_results = pd.DataFrame(all_summaries)
    print("\n" + "="*50)
    print("RL EVALUATION SUMMARY")
    print("="*50)
    print(df_results[["agent", "success_rate", "sla_breach_rate", "avg_reward", "avg_steps"]].to_string(index=False))
    print("="*50)
    print(f"Full reports saved to {REPORT_DIR}")

if __name__ == "__main__":
    main()
