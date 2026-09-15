import numpy as np
from typing import Dict, Any

from marl.environment.resource import ResourceState
from marl.environment.task import TaskRepresentation

# Status: Implemented

def compute_latency_reward(env_info: Dict[str, Any]) -> float:
    """
    Penalizes high total completion time relative to deadline.
    [LOCAL]
    """
    task = env_info.get('allocated_task')
    if not task:
        return 0.0
        
    total_completion_time = env_info.get('total_completion_time', 0.0)
    deadline = getattr(task, 'deadline', 1.0) # avoid div by zero, default to 1.0
    if deadline == 0:
        deadline = 1e-5
        
    if env_info.get('sla_violated', False) or total_completion_time > deadline:
        return -(total_completion_time / deadline)
    else:
        return max(0.0, 1.0 - total_completion_time / deadline)

def compute_energy_reward(env_info: Dict[str, Any]) -> float:
    """
    Penalizes energy consumption.
    [LOCAL]
    """
    energy_consumed = env_info.get('energy_consumed', 0.0)
    max_energy_scale = 1000.0
    return -(energy_consumed / max_energy_scale)

def compute_utilization_reward(env_info: Dict[str, Any]) -> float:
    """
    Rewards efficient utilization without overload.
    [GLOBAL]
    """
    avg_util = env_info.get('system_avg_utilization', 0.0)
    if 0.4 <= avg_util <= 0.8:
        return 1.0
    elif avg_util < 0.4:
        return avg_util / 0.4
    else:
        return max(0.0, 1.0 - (avg_util - 0.8) / 0.2)

def compute_load_balance_reward(env_info: Dict[str, Any]) -> float:
    """
    Penalizes variance in resource utilizations.
    [GLOBAL]
    """
    utils = env_info.get('all_utilizations', [])
    if not utils:
        return 0.0
    var = np.var(utils)
    scale_factor = 10.0
    reward = -float(var) * scale_factor
    return max(-1.0, min(0.0, reward))

def compute_sla_reward(env_info: Dict[str, Any]) -> float:
    """
    Strong penalty for SLA violations, priority-weighted.
    [LOCAL]
    """
    task_priority = env_info.get('task_priority', 1)
    sla_penalty_scale = 0.5
    
    if env_info.get('sla_violated', False):
        return -float(task_priority) * sla_penalty_scale
    else:
        slack = env_info.get('sla_slack', 0.0)
        task = env_info.get('allocated_task')
        deadline = getattr(task, 'deadline', 1.0) if task else 1.0
        if deadline == 0:
            deadline = 1e-5
        return min(1.0, (slack / deadline) * task_priority * 0.1)

def compute_queue_reward(env_info: Dict[str, Any]) -> float:
    """
    Penalizes excessive queue waiting time.
    [LOCAL]
    """
    length = env_info.get('local_queue_length', 0)
    max_queue_scale = 20.0
    reward = -(length / max_queue_scale)
    return max(-1.0, min(0.0, reward))

def compute_throughput_reward(env_info: Dict[str, Any]) -> float:
    """
    Rewards completed tasks.
    [GLOBAL]
    """
    completed = env_info.get('total_completed_tasks', 0)
    total_this_step = max(1, env_info.get('total_tasks_this_step', 1))
    reward = completed / total_this_step
    return max(0.0, min(1.0, float(reward)))



def compute_action_penalty(env_info: dict) -> float:
    """
    Applies explicit penalties for agent's local scheduling actions (DEFER/REJECT).
    [LOCAL]
    """
    if env_info.get('action_was_reject', False):
        return -1.0
    if env_info.get('action_was_defer', False):
        return -0.1
    return 0.0


REWARD_COMPONENTS = {
    'action_penalty': {'fn': compute_action_penalty, 'scope': 'local'},
    'latency': {'fn': compute_latency_reward, 'scope': 'local'},
    'energy': {'fn': compute_energy_reward, 'scope': 'local'},
    'utilization': {'fn': compute_utilization_reward, 'scope': 'global'},
    'load_balance': {'fn': compute_load_balance_reward, 'scope': 'global'},
    'sla': {'fn': compute_sla_reward, 'scope': 'local'},
    'queue': {'fn': compute_queue_reward, 'scope': 'local'},
    'throughput': {'fn': compute_throughput_reward, 'scope': 'global'},
}