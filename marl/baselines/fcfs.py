from typing import Dict, Any
from marl.baselines.base_scheduler import Scheduler

class FCFSScheduler(Scheduler):
    """First-Come-First-Served: assigns task to the first feasible resource.
    
    For each agent, iterates through its resources in order.
    Assigns to the first resource that can execute the current task.
    If none feasible, defers.
    
    Status: Baseline
    """
    def __init__(self):
        super().__init__(name='fcfs')

    def select_actions(self, env: Any) -> Dict[int, int]:
        actions = {}
        for agent_id in env.agent_ids:
            mask = env.get_action_mask(agent_id)
            # Find first valid action excluding defer (which is the last index)
            action = len(mask) - 2 # Default to defer
            for i in range(len(mask) - 2):
                if mask[i] == 1:
                    action = i
                    break
            actions[agent_id] = action
        return actions
