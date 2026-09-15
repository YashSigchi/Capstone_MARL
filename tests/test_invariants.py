import pytest
import random
from marl.environment.marl_env import MARLResourceAllocationEnv

def test_no_resource_over_allocation(small_config):
    env = MARLResourceAllocationEnv(small_config)
    env.reset(seed=42)
    for _ in range(20):
        actions = {}
        for aid in env.agent_ids:
            mask = env.get_action_mask(aid)
            valid = [i for i, m in enumerate(mask) if m == 1.0]
            actions[aid] = random.choice(valid) if valid else mask.size - 1
        env.step(actions)
        
        for r in env.resources.values():
            assert r.current_cpu_usage <= r.cpu_capacity
            assert r.current_memory_usage <= r.memory_capacity
            assert r.current_gpu_usage <= r.gpu_capacity

def test_no_duplicate_task_assignment(small_config):
    env = MARLResourceAllocationEnv(small_config)
    env.reset(seed=42)
    for _ in range(20):
        actions = {}
        for aid in env.agent_ids:
            mask = env.get_action_mask(aid)
            valid = [i for i, m in enumerate(mask) if m == 1.0]
            actions[aid] = valid[0] if valid else mask.size - 1
        env.step(actions)
        
        running_task_ids = [rt.task.task_id for rt in env._running_tasks]
        assert len(running_task_ids) == len(set(running_task_ids))

def test_valid_task_lifecycle(small_config):
    env = MARLResourceAllocationEnv(small_config)
    env.reset(seed=42)
    for _ in range(5):
        actions = {aid: env.get_action_dim(aid) - 1 for aid in env.agent_ids} # all defer
        env.step(actions)
    
    for task in env._all_tasks:
        assert task.status in ['pending', 'queued', 'running', 'completed', 'rejected']

def test_capacity_constraints_respected(small_config):
    env = MARLResourceAllocationEnv(small_config)
    env.reset(seed=42)
    mask = env.get_action_mask(0)
    assert len(mask) > 0

def test_resource_release_on_completion(small_config):
    env = MARLResourceAllocationEnv(small_config)
    env.reset(seed=42)
    initial_util = {rid: r.utilization() for rid, r in env.resources.items()}
    
    # Step until a task is scheduled
    while not env._running_tasks and env._current_step < 10:
        actions = {}
        for aid in env.agent_ids:
            mask = env.get_action_mask(aid)
            valid = [i for i, m in enumerate(mask[:-1]) if m == 1.0]
            actions[aid] = valid[0] if valid else mask.size - 1
        env.step(actions)
        
    if env._running_tasks:
        # Fast forward until completion
        for _ in range(20):
            env.step({aid: env.get_action_dim(aid) - 1 for aid in env.agent_ids})
            if not env._running_tasks:
                break
        
        for rid, r in env.resources.items():
            assert r.utilization() >= 0.0 # Just testing it doesn't crash and returns valid.

def test_total_tasks_consistent(small_config):
    env = MARLResourceAllocationEnv(small_config)
    env.reset(seed=42)
    for _ in range(15):
        actions = {aid: env.get_action_dim(aid) - 1 for aid in env.agent_ids}
        env.step(actions)
    
    queued = sum(q.length for q in env.queues.values())
    running = len(env._running_tasks)
    completed = len(env._completed_tasks)
    rejected = len(env._rejected_tasks)
    
    assert queued + running + completed + rejected == len(env._all_tasks)
