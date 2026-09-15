import pytest
import numpy as np
from marl.environment.marl_env import MARLResourceAllocationEnv

def test_reset_returns_correct_structure(small_config):
    env = MARLResourceAllocationEnv(small_config)
    obs, infos = env.reset(seed=42)
    assert set(obs.keys()) == set(env.agent_ids)
    assert set(infos.keys()) == set(env.agent_ids)
    for agent_id, o in obs.items():
        assert o.shape == (env.get_obs_dim(),)
        assert isinstance(o, np.ndarray)

def test_reset_clears_state(small_config):
    env = MARLResourceAllocationEnv(small_config)
    env.reset(seed=42)
    env.step({0: 0, 1: 0})
    env.reset(seed=43)
    assert env._current_step == 0
    assert len(env._running_tasks) == 0
    assert len(env._completed_tasks) == 0
    for r in env.resources.values():
        assert r.current_cpu_usage == 0.0

def test_step_returns_correct_structure(small_config):
    env = MARLResourceAllocationEnv(small_config)
    env.reset(seed=42)
    obs, rewards, terminated, truncated, infos = env.step({0: 2, 1: 2})
    assert set(obs.keys()) == set(env.agent_ids)
    assert set(rewards.keys()) == set(env.agent_ids)
    assert set(terminated.keys()) == set(env.agent_ids)
    assert set(truncated.keys()) == set(env.agent_ids)
    assert set(infos.keys()) == set(env.agent_ids)

def test_step_terminated_signal(small_config):
    small_config['episode_length'] = 5
    small_config['max_tasks_per_episode'] = 1
    small_config['task_arrival']['rate'] = 1
    env = MARLResourceAllocationEnv(small_config)
    env.reset(seed=42)
    for _ in range(10):
        obs, rewards, term, trunc, infos = env.step({0: env.get_action_dim(0) - 1, 1: env.get_action_dim(1) - 1})
        if any(term.values()) or any(trunc.values()):
            break
    assert any(term.values()) or any(trunc.values())

def test_task_arrival(small_config):
    env = MARLResourceAllocationEnv(small_config)
    env.reset(seed=42)
    total_queue_len = sum(q.length for q in env.queues.values())
    assert total_queue_len > 0

def test_task_completion(small_config):
    env = MARLResourceAllocationEnv(small_config)
    env.reset(seed=42)
    for _ in range(50):
        actions = {}
        for aid in env.agent_ids:
            mask = env.get_action_mask(aid)
            valid_actions = [i for i, m in enumerate(mask) if m == 1.0]
            actions[aid] = valid_actions[0] if valid_actions else mask.size - 1
        env.step(actions)
        if len(env._completed_tasks) > 0:
            break
    assert len(env._completed_tasks) > 0

def test_resource_state_changes(small_config):
    env = MARLResourceAllocationEnv(small_config)
    env.reset(seed=42)
    initial_util = env.resources['R1'].utilization()
    
    actions = {0: 0, 1: 2} # 2 is defer
    env.step(actions)
    if any(rt.resource_id == 'R1' for rt in env._running_tasks):
        assert env.resources['R1'].utilization() > initial_util

def test_observation_dimension_consistency(small_config):
    env = MARLResourceAllocationEnv(small_config)
    obs, _ = env.reset(seed=42)
    dim = env.get_obs_dim()
    for o in obs.values():
        assert o.shape[0] == dim
    
    obs, _, _, _, _ = env.step({0: 0, 1: 0})
    for o in obs.values():
        assert o.shape[0] == dim

def test_action_space_correct(small_config):
    env = MARLResourceAllocationEnv(small_config)
    assert env.action_space(0).n == 4
    assert env.get_action_dim(0) == 4

def test_global_state_dimension(small_config):
    env = MARLResourceAllocationEnv(small_config)
    env.reset(seed=42)
    global_state = env.get_global_state()
    assert global_state.shape == (env.get_global_state_dim(),)
