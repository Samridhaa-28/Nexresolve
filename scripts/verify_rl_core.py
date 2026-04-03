import sys
import os
import torch
import numpy as np

# Ensure root dir is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rl.replay_buffer import ReplayBuffer
from rl.level1_dqn import Level1Agent
from rl.level2_dqn import Level2Agent
from rl.training_utils import epsilon_scheduler, bandit_mix_scheduler, TrainingLogger
from rl.action_masking import get_action_mask
from rl.action_space import Strategy, get_action_count

def test_rl_core():
    print("--- 🔬 Starting RL Core Verification ---")
    
    # 1. State and Mask Initialization
    state = np.random.rand(37).astype(np.float32)
    # Mock some values that trigger masking in action_masking.py
    state[6] = 0.5  # sla_remaining
    state[14] = 0   # urgent_flag
    state[21] = 0.8 # completeness_score
    state[23] = 1   # needs_clarify
    state[27] = 0.1 # frustration
    state[31] = 0.5 # max_sim
    state[35] = 1   # tier1
    state[36] = 0   # tier2
    
    mask = get_action_mask(state)
    print(f"Action mask sum: {np.sum(mask)} / {get_action_count()}")

    # 2. Replay Buffer
    buffer = ReplayBuffer(capacity=100)
    for i in range(10):
        buffer.add(state, 1, 1.0, state, False)
    print(f"Buffer size: {len(buffer)}")
    batch = buffer.sample(batch_size=5)
    print(f"Sampled batch shapes: {batch[0].shape}, {batch[1].shape}, {batch[2].shape}")

    # 3. Schedulers
    eps = epsilon_scheduler(2500)
    print(f"Epsilon at step 2500: {eps:.4f} (expected ~0.525)")
    bm = bandit_mix_scheduler(1000)
    print(f"Bandit mix at step 1000: {bm:.4f} (expected ~0.45)")

    # 4. Level 1 Agent
    l1 = Level1Agent()
    strategy_idx = l1.select_action(state, mask)
    print(f"Selected strategy: {Strategy(strategy_idx).name}")
    
    # 5. Level 2 Agent
    l2 = Level2Agent()
    global_action = l2.select_action(state, strategy_idx, mask)
    print(f"Selected global action: {global_action}")

    # 6. Update Test
    l1_loss = l1.update(batch)
    print(f"Level 1 update loss: {l1_loss:.4f}")
    
    l2_losses = l2.update(batch)
    print(f"Level 2 update losses: {l2_losses}")
    # 7. Model Saving/Loading
    save_dir = "tmp/rl_test_models"
    os.makedirs(save_dir, exist_ok=True)
    l1.save(os.path.join(save_dir, "l1.pth"))
    l2.save_all(save_dir)
    print("Models saved successfully.")
    
    l1.load(os.path.join(save_dir, "l1.pth"))
    l2.load_all(save_dir)
    print("Models loaded successfully.")

    print("--- ✅ RL Core Verification Complete ---")

if __name__ == "__main__":
    test_rl_core()
