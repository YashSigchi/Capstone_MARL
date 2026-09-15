from types import SimpleNamespace
import yaml
from marl.reward.reward_engine import RewardEngine
from marl.reward.reward_components import REWARD_COMPONENTS
import numpy as np

with open('configs/reward.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Disable normalization for clear raw inspection
config['normalization'] = 'running_mean_std'
engine = RewardEngine(config)

# Dummy environment info for three actions:
# Task: 50 work units, deadline 20. SLA class 'bronze' (priority 1).

# Action 1: ALLOCATE (Successful, takes 5 steps to complete, consumes 100 energy)
alloc_info = {
    'action_was_reject': False,
    'action_was_defer': False,
    
    'allocated_task': SimpleNamespace(deadline=20.0), # just a truthy value
    'total_completion_time': 30.0, # wait 0 + net 0 + exec 5
    'task_deadline': 20.0,
    'energy_consumed': 100.0,
    'sla_violated': True,
    'task_priority': 1,
    'local_queue_length': 0,
    'total_completed_tasks': 1,
    'total_energy': 100.0,
    'system_avg_utilization': 0.5,
    'resource_imbalance': 0.1,
    'total_sla_violations': 0,
    'total_tasks_processed': 1
}

# Action 2: DEFER
defer_info = {
    'action_was_reject': False,
    'action_was_defer': True,
    'allocated_task': None,
    'local_queue_length': 1,
    'total_completed_tasks': 0,
    'total_energy': 0.0,
    'system_avg_utilization': 0.0,
    'resource_imbalance': 0.0,
    'total_sla_violations': 0,
    'total_tasks_processed': 0
}

# Action 3: REJECT
reject_info = {
    'action_was_reject': True,
    'action_was_defer': False,
    'allocated_task': None,
    'local_queue_length': 0,
    'total_completed_tasks': 0,
    'total_energy': 0.0,
    'system_avg_utilization': 0.0,
    'resource_imbalance': 0.0,
    'total_sla_violations': 0, # wait, does reject count as SLA violation?
    'total_tasks_processed': 1
}

print("=== REWARD AUDIT ===")
for name, info in [("ALLOCATE", alloc_info), ("DEFER", defer_info), ("REJECT", reject_info)]:
    print(f"\nAction: {name}")
    res = engine.compute(info)
    for comp, val in res.components.items():
        print(f"  {comp}: {val:.4f} (weighted: {res.weighted_components[comp]:.4f})")
    print(f"  TOTAL: {res.total:.4f} (Local: {res.local:.4f}, Global: {res.global_:.4f})")

