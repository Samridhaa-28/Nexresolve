import sys
import os

print("Hello from minimal test")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from rl.action_space import ACTIONS
    print(f"Action space loaded: {len(ACTIONS)} actions")
    
    from rl.action_masking import get_action_mask
    print("Action masking loaded")
    
    from rl.reward import compute_reward
    print("Reward loaded")
    
    # Try importing environment - this might be the slow part
    print("Attempting to import environment...")
    from rl.environment import NexResolveEnv
    print("Environment loaded")
    
except Exception as e:
    print(f"Error: {e}")
