from typing import Dict, Any
from marl.baselines.base_scheduler import Scheduler
import numpy as np

class METScheduler(Scheduler):
    """Minimum Execution Time: assigns task to the resource that would finish fastest.
    
    For each agent, among feasible resources, picks the one where
    task.work_units / resource.processing_speed is minimized.
    
    Status: Baseline
    """
    def __init__(self):
        super().__init__(name='met')

    def select_actions(self, env: Any) -> Dict[int, int]:
        actions = {}
        for agent_id in env.agent_ids:
            mask = env.get_action_mask(agent_id)
            task = env.get_current_task(agent_id)
            resources = env.get_agent_resources(agent_id)
            
            best_action = len(mask) - 2 # defer
            best_time = float('inf')
            
            if task is not None:
                for i in range(len(mask) - 2):
                    if mask[i] == 1:
                        res = resources[i]
                        time = task.work_units / max(res.processing_speed, 1e-6)
                        if time < best_time:
                            best_time = time
                            best_action = i
                            
            actions[agent_id] = best_action
            
        return actions
