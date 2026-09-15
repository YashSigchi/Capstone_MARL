import pytest
import numpy as np
from marl.training.rollout_buffer import RolloutBuffer

def test_gae_numerical_validation():
    # Setup a small buffer: 1 agent, 3 steps
    num_steps = 3
    num_agents = 1
    action_dim = 2
    gamma = 0.99
    gae_lambda = 0.95
    
    buffer = RolloutBuffer(num_agents, num_steps, obs_dim=4, global_state_dim=4, action_dim=action_dim)
    
    # We will simulate the following scenario:
    # Step 0: V(s0) = 0.5, r0 = 1.0, terminated = False, truncated = False
    # Step 1: V(s1) = 0.7, r1 = 1.0, terminated = False, truncated = False
    # Step 2: V(s2) = 0.9, r2 = 1.0, terminated = False, truncated = True
    
    # The last_values (bootstrap for step 2) will be V(s3) = 1.0
    last_values = {0: 1.0}
    
    buffer.insert(0, 0, np.zeros(4), np.zeros(4), 0, 0.0, 1.0, 0.5, False, False, np.ones(2))
    buffer.insert(1, 0, np.zeros(4), np.zeros(4), 0, 0.0, 1.0, 0.7, False, False, np.ones(2))
    buffer.insert(2, 0, np.zeros(4), np.zeros(4), 0, 0.0, 1.0, 0.9, False, True, np.ones(2))
    
    buffer.compute_returns_and_advantages(gamma, gae_lambda, last_values)
    
    # Let's manually compute GAE:
    # Step 2:
    # V(s3) = 1.0 (since not terminated, bootstrap is used despite truncation)
    # delta_2 = r_2 + gamma * V(s3) - V(s2)
    # delta_2 = 1.0 + 0.99 * 1.0 - 0.9 = 1.09
    # A_2 = delta_2 = 1.09
    
    # Step 1:
    # delta_1 = r_1 + gamma * V(s2) - V(s1)
    # delta_1 = 1.0 + 0.99 * 0.9 - 0.7 = 1.891 - 0.7 = 1.191
    # A_1 = delta_1 + gamma * lambda * A_2
    # A_1 = 1.191 + 0.99 * 0.95 * 1.09 = 1.191 + 1.025145 = 2.216145
    
    # Step 0:
    # delta_0 = r_0 + gamma * V(s1) - V(s0)
    # delta_0 = 1.0 + 0.99 * 0.7 - 0.5 = 1.693 - 0.5 = 1.193
    # A_0 = delta_0 + gamma * lambda * A_1
    # A_0 = 1.193 + 0.99 * 0.95 * 2.216145 = 1.193 + 2.084285 = 3.277285
    
    expected_adv = np.array([3.277285, 2.216145, 1.09], dtype=np.float32)
    expected_ret = expected_adv + np.array([0.5, 0.7, 0.9], dtype=np.float32)
    
    np.testing.assert_allclose(buffer.advantages[:, 0], expected_adv, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(buffer.returns[:, 0], expected_ret, rtol=1e-5, atol=1e-5)
    
    # Now simulate a termination at step 1:
    buffer2 = RolloutBuffer(num_agents, num_steps, obs_dim=4, global_state_dim=4, action_dim=action_dim)
    buffer2.insert(0, 0, np.zeros(4), np.zeros(4), 0, 0.0, 1.0, 0.5, False, False, np.ones(2))
    buffer2.insert(1, 0, np.zeros(4), np.zeros(4), 0, 0.0, 1.0, 0.7, True, False, np.ones(2))
    buffer2.insert(2, 0, np.zeros(4), np.zeros(4), 0, 0.0, 0.0, 0.0, True, False, np.ones(2))
    
    buffer2.compute_returns_and_advantages(gamma, gae_lambda, {0: 0.0})
    
    # Step 2: delta_2 = 0 + 0 - 0 = 0. A_2 = 0
    # Step 1: terminated=True -> next_non_terminal = 0.0
    # delta_1 = r_1 + 0 - V(s1) = 1.0 - 0.7 = 0.3
    # A_1 = 0.3 + 0 = 0.3
    # Step 0: terminated=False -> next_non_terminal = 1.0
    # delta_0 = r_0 + gamma * V(s1) - V(s0) = 1.0 + 0.99 * 0.7 - 0.5 = 1.193
    # A_0 = 1.193 + gamma * lambda * A_1 = 1.193 + 0.99 * 0.95 * 0.3 = 1.193 + 0.28215 = 1.47515
    
    expected_adv2 = np.array([1.47515, 0.3, 0.0], dtype=np.float32)
    np.testing.assert_allclose(buffer2.advantages[:, 0], expected_adv2, rtol=1e-5, atol=1e-5)
