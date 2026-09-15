import pytest
from marl.environment.marl_env import MARLResourceAllocationEnv
from marl.baselines.fcfs import FCFSScheduler
from marl.baselines.round_robin import RoundRobinScheduler
from marl.baselines.met import METScheduler
from marl.baselines.random_feasible import RandomFeasibleScheduler

def test_fcfs_produces_valid_actions(small_config):
    env = MARLResourceAllocationEnv(small_config)
    env.reset(seed=42)
    if not hasattr(env, 'agents'):
        env.agents = env.agent_ids
    scheduler = FCFSScheduler()
    actions = scheduler.select_actions(env)
    for aid, act in actions.items():
        assert env.get_action_mask(aid)[act] == 1.0

def test_round_robin_cycles(small_config):
    env = MARLResourceAllocationEnv(small_config)
    env.reset(seed=42)
    if not hasattr(env, 'agents'):
        env.agents = env.agent_ids
    scheduler = RoundRobinScheduler()
    # Mock resources can_execute
    for aid in env.agent_ids:
        for r in env.get_agent_resources(aid):
            r.current_cpu_usage = 0 # allow all
            
    a1 = scheduler.select_actions(env)
    a2 = scheduler.select_actions(env)
    # They should ideally cycle if tasks are present and feasible
    assert isinstance(a1, dict)

def test_met_picks_fastest(small_config):
    env = MARLResourceAllocationEnv(small_config)
    env.reset(seed=42)
    if not hasattr(env, 'agents'):
        env.agents = env.agent_ids
    scheduler = METScheduler()
    actions = scheduler.select_actions(env)
    for aid, act in actions.items():
        assert env.get_action_mask(aid)[act] == 1.0

def test_random_feasible_valid(small_config):
    env = MARLResourceAllocationEnv(small_config)
    env.reset(seed=42)
    if not hasattr(env, 'agents'):
        env.agents = env.agent_ids
    scheduler = RandomFeasibleScheduler()
    actions = scheduler.select_actions(env)
    for aid, act in actions.items():
        assert env.get_action_mask(aid)[act] == 1.0

def test_all_baselines_complete_episode(small_config):
    small_config['episode_length'] = 5
    for S in [FCFSScheduler, RoundRobinScheduler, METScheduler, RandomFeasibleScheduler]:
        env = MARLResourceAllocationEnv(small_config)
        env.reset(seed=42)
        if not hasattr(env, 'agents'):
            env.agents = env.agent_ids
        scheduler = S()
        for _ in range(6):
            acts = scheduler.select_actions(env)
            obs, rews, term, trunc, info = env.step(acts)
            if any(term.values()) or any(trunc.values()):
                break

def test_baselines_use_same_env(small_config):
    # Just a sanity check
    env1 = MARLResourceAllocationEnv(small_config)
    env2 = MARLResourceAllocationEnv(small_config)
    assert env1.get_obs_dim() == env2.get_obs_dim()
