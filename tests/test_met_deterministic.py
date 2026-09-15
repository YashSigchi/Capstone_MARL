import pytest
import numpy as np
import yaml
from marl.environment.marl_env import MARLResourceAllocationEnv
from marl.baselines.fcfs import FCFSScheduler
from marl.baselines.met import METScheduler

def test_met_vs_fcfs():
    with open('configs/environment.yaml', 'r') as f:
        config = yaml.safe_load(f)
    with open('configs/marl.yaml', 'r') as f:
        marl_config = yaml.safe_load(f)
    config.update(marl_config)
    
    env = MARLResourceAllocationEnv(config)
    env.reset(seed=42)
    
    resources = env.get_agent_resources(0)
    r0_id = resources[0].resource_id
    r1_id = resources[1].resource_id
    
    # Make sure both have enough capacity
    env.resources[r0_id].cpu_capacity = 1000.0
    env.resources[r1_id].cpu_capacity = 1000.0
    env.resources[r0_id].memory_capacity = 1000.0
    env.resources[r1_id].memory_capacity = 1000.0
    env.resources[r0_id].gpu_capacity = 1000.0
    env.resources[r1_id].gpu_capacity = 1000.0
    
    env.resources[r0_id].processing_speed = 1.0
    env.resources[r1_id].processing_speed = 2.0
    
    fcfs = FCFSScheduler()
    actions = fcfs.select_actions(env)
    assert actions[0] == 0, f"FCFS should pick first resource, picked {actions[0]}"
    
    met = METScheduler()
    actions = met.select_actions(env)
    assert actions[0] == 1, f"MET should pick the fastest resource (index 1), picked {actions[0]}"

