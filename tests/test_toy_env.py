import pytest
import numpy as np
import yaml
from marl.environment.marl_env import MARLResourceAllocationEnv
from marl.environment.task import TaskRepresentation
from marl.reward.reward_engine import RewardEngine

def test_toy_env():
    # 1 agent, 2 resources
    with open('configs/environment.yaml', 'r') as f:
        config = yaml.safe_load(f)
    with open('configs/marl.yaml', 'r') as f:
        marl_config = yaml.safe_load(f)
    config.update(marl_config)
    config['max_steps'] = 10
    config['task_arrival']['rate'] = 0.0
    
    env = MARLResourceAllocationEnv(config)
    obs, _ = env.reset(seed=42)
    
    # Empty queue, add custom tasks
    env.queues[0].clear()
    
    t1 = TaskRepresentation(
        task_id='t1', cpu_req=1.0, memory_req=1.0, gpu_req=0.0,
        work_units=5.0, deadline=10.0, priority=1, data_size=0.0,
        communication_req=0.0, dependencies=[], arrival_time=0.0, sla_class='bronze'
    )
    t2 = TaskRepresentation(
        task_id='t2', cpu_req=1.0, memory_req=1.0, gpu_req=0.0,
        work_units=10.0, deadline=15.0, priority=1, data_size=0.0,
        communication_req=0.0, dependencies=[], arrival_time=0.0, sla_class='bronze'
    )
    t1.status = 'queued'
    t2.status = 'queued'
    env.queues[0].enqueue(t1)
    env.queues[0].enqueue(t2)
    env._all_tasks.clear()
    env._all_tasks.append(t1)
    env._all_tasks.append(t2)
    
    # Step 1: Allocate t1 to resource 0
    obs, rewards, term, trunc, info = env.step({0: 0})
    assert t1.status == 'running'
    assert env.queues[0].peek() == t2
    
    # Step 2: Reject t2 (action = 3 for agent with 2 resources)
    obs, rewards, term, trunc, info = env.step({0: 3})
    assert t2.status == 'rejected'
    assert env.queues[0].is_empty
    assert info[0]['reward_info']['action_was_reject'] == True
    
    # Step 3, 4, 5, 6, 7: empty actions (defer since no tasks)
    env.step({0: 2})
    env.step({0: 2})
    env.step({0: 2})
    env.step({0: 2})
    
    assert t1.status == 'completed'
    assert env._makespan >= 5.0
    
    # Verify reward logic via engine
    with open('configs/reward.yaml', 'r') as f:
        reward_config = yaml.safe_load(f)
    
    reward_engine = RewardEngine(reward_config)
    rew_result = reward_engine.compute(info[0]['reward_info'])
    assert isinstance(rew_result.total, float)

