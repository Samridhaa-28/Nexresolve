

import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rl.action_space import ACTION_TO_INDEX, INDEX_TO_ACTION, get_action_count
from rl.action_masking import get_action_mask, get_valid_actions
from rl.reward import compute_reward
from rl.environment import NexResolveEnv
from rl.bandit_policy import LinUCB, ThompsonSampling

def test_action_masking():
    print("\n[Test] Action Masking...")
    
    # State with completeness < 0.5 (should disable SUGGEST)
    state_no_suggest = np.zeros(37)
    state_no_suggest[21] = 0.4 
    mask = get_action_mask(state_no_suggest)
    suggest_indices = [i for i, name in INDEX_TO_ACTION.items() if "suggest" in name]
    for idx in suggest_indices:
        assert mask[idx] == 0.0, f"Action {INDEX_TO_ACTION[idx]} should be masked"
    print("  ✓ completeness_score < 0.5 disables suggest")

    # State with frustration > 0.7 (should disable CLARIFY)
    state_frustrated = np.zeros(37)
    state_frustrated[27] = 0.8
    state_frustrated[21] = 1.0 # Enable suggest
    state_frustrated[6] = 1.0 # High SLA
    mask = get_action_mask(state_frustrated)
    clarify_indices = [i for i, name in INDEX_TO_ACTION.items() if "ask_" in name]
    for idx in clarify_indices:
        assert mask[idx] == 0.0, f"Action {INDEX_TO_ACTION[idx]} should be masked"
    print("  ✓ frustration_level > 0.7 disables clarify")

    # State with low SLA (should disable CLARIFY)
    state_low_sla = np.zeros(37)
    state_low_sla[6] = 0.1
    state_low_sla[23] = 1.0 # Needs clarify
    mask = get_action_mask(state_low_sla)
    for idx in clarify_indices:
        assert mask[idx] == 0.0, f"Action {INDEX_TO_ACTION[idx]} should be masked"
    print("  ✓ sla_remaining_norm < 0.2 disables clarify")

def test_reward_function():
    print("\n[Test] Reward Function...")
    s1 = np.zeros(37)
    s2 = np.zeros(37)
    
    # Resolution + SLA
    s2[30] = 1.0 # resolved
    s2[5] = 0.0  # no breach
    r = compute_reward(s1, ACTION_TO_INDEX["suggest_top1"], s2)
    # Expected: +10 (res) + 5 (sla) - 1 (step) + 2 (correct suggest since sim is 0 but it's success? 
    # wait, reward.py checks next_state[31] >= 0.85 for correct suggest. Here it is 0.0)
    # Correct suggest: +2 if next_state[30]==1 and next_state[31] >= 0.85. 
    # So r = 10 + 5 - 1 = 14.
    assert r == 14.0, f"Expected 14.0, got {r}"
    print("  ✓ Resolution + SLA reward computed correctly")

    # SLA Breach
    s1_breach = np.zeros(37)
    s2_breach = np.zeros(37)
    s2_breach[5] = 1.0 # breach
    r = compute_reward(s1_breach, ACTION_TO_INDEX["route_bug"], s2_breach)
    # Expected: -1 (step) - 10 (breach) = -11
    assert r == -11.0, f"Expected -11.0, got {r}"
    print("  ✓ SLA Breach penalty computed correctly")

def test_environment_transitions():
    print("\n[Test] Environment Transitions...")
    env = NexResolveEnv()
    s = env.reset(ticket_idx=0)
    
    # Test CLARIFY: ask_version
    action_id = ACTION_TO_INDEX["ask_version"]
    s_next, r, d, info = env.step(action_id)
    assert s_next[15] == 1.0, "has_version should be 1.0"
    assert s_next[4] == s[4] + 1.0, "turn_count should increase"
    print("  ✓ CLARIFY updates missing flags and turn count")

    # Test ROUTE: route_bug
    s_before_route = s_next.copy()
    action_id = ACTION_TO_INDEX["route_bug"]
    s_next, r, d, info = env.step(action_id)
    assert s_next[22] == s_before_route[22] + 1.0, "reassignment_count should increase"
    print("  ✓ ROUTE updates reassignment count")

    # Test Determinism
    env.reset(ticket_idx=5)
    s1, _, _, _ = env.step(0)
    env.reset(ticket_idx=5)
    s2, _, _, _ = env.step(0)
    assert np.array_equal(s1, s2), "Transitions should be deterministic"
    print("  ✓ Environment transitions are deterministic")

def test_bandit_policies():
    print("\n[Test] Bandit Policies...")
    context = np.random.rand(37)
    
    # LinUCB
    linucb = LinUCB()
    a = linucb.select_action(context)
    linucb.update(context, a, 1.0)
    print("  ✓ LinUCB select and update")
    
    # Thompson Sampling
    ts = ThompsonSampling()
    a = ts.select_action(context)
    ts.update(context, a, 1.0)
    print("  ✓ Thompson Sampling select and update")
def test_episode_run():
    print("\n[Test] Episode Run...")
    env = NexResolveEnv()
    s = env.reset()
    
    for _ in range(10):
        mask = get_action_mask(s)
        valid = np.where(mask == 1)[0]
        a = valid[0]
        s, r, d, _ = env.step(a)
        if d:
            break
            
    print("  ✓ Episode runs without crash")

if __name__ == "__main__":
    try:
        test_action_masking()
        test_reward_function()
        test_environment_transitions()
        test_bandit_policies()
        test_episode_run()
        print("\n[SUCCESS] All RL components verified!")
    except Exception as e:
        print(f"\n[FAILURE] {e}")
        sys.exit(1)
