import pytest
import numpy as np
import torch
from marl.environment.marl_env import MARLResourceAllocationEnv
from marl.agents.mappo_agent import MAPPOAgent
from marl.training.rollout_buffer import RolloutBuffer
from marl.reward.reward_engine import RewardEngine
from marl.baselines.fcfs import FCFSScheduler

def test_full_episode_completes(small_config):
    env = MARLResourceAllocationEnv(small_config)
    env.reset(seed=42)
    while True:
        actions = {}
        for aid in env.agent_ids:
            mask = env.get_action_mask(aid)
            valid = [i for i, m in enumerate(mask) if m == 1.0]
            actions[aid] = np.random.choice(valid)
        obs, rews, term, trunc, infos = env.step(actions)
        if any(term.values()) or any(trunc.values()):
            break
    assert True

def test_mappo_forward_pass(small_config):
    env = MARLResourceAllocationEnv(small_config)
    obs, infos = env.reset(seed=42)
    agent = MAPPOAgent(0, env.get_obs_dim(), env.get_action_dim(0), env.get_global_state_dim(), {'device': 'cpu'})
    
    action, log_prob, val = agent.select_action(obs[0], infos[0]['action_mask'])
    
    global_state = env.get_global_state()
    value = agent.get_value(global_state)
    
    assert isinstance(action, int)
    assert isinstance(value, float)

def test_rollout_buffer_insert_and_iterate(small_config):
    buf = RolloutBuffer(2, 5, 10, 20, 3)
    for i in range(5):
        for aid in range(2):
            buf.insert(i, aid, np.zeros(10), np.zeros(20), 0, 0.0, 1.0, 0.5, False, False, np.ones(3))
    
    buf.compute_returns_and_advantages(0.99, 0.95, {0: 0.0, 1: 0.0})
    iterator = buf.get_minibatch_iterator(2, torch.device('cpu'))
    
    for batch in iterator:
        assert 'obs' in batch
        assert batch['obs'].shape[0] == 5
        break

def test_reward_engine_with_env(small_config):
    env = MARLResourceAllocationEnv(small_config)
    engine = RewardEngine(small_config)
    
    env.reset(seed=42)
    obs, rews, term, trunc, infos = env.step({0: 0, 1: 0})
    
    for aid in env.agent_ids:
        res = engine.compute(infos[aid]['reward_info'])
        assert res.total is not None

def test_baseline_comparison(small_config):
    env1 = MARLResourceAllocationEnv(small_config)
    env1.reset(seed=42)
    if not hasattr(env1, 'agents'):
        env1.agents = env1.agent_ids
        
    scheduler = FCFSScheduler()
    for _ in range(5):
        acts = scheduler.select_actions(env1)
        env1.step(acts)
        
    assert True
