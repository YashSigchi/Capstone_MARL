from typing import Dict, Any, List
import numpy as np
from marl.baselines.base_scheduler import Scheduler
from marl.environment.marl_env import MARLResourceAllocationEnv

class GreedyEnergyScheduler(Scheduler):
    """
    Allocates tasks to the feasible resource that is estimated to consume the minimum energy.
    """
    def __init__(self):
        super().__init__(name='greedy_energy')

    def select_actions(self, env_ref: MARLResourceAllocationEnv):
        actions = {}
        for aid in env_ref.agent_ids:
            mask = env_ref.get_action_mask(aid)
            valid_acts = np.where(mask == 1.0)[0]
            
            if len(valid_acts) == 0:
                actions[aid] = len(mask) - 2 # DEFER fallback
                continue
                
            active_task = env_ref.queues[aid].peek()
            if active_task is None:
                actions[aid] = len(mask) - 2
                continue
                
            cluster = env_ref._get_cluster(aid)
            c_resources = cluster.get_resources(env_ref.resources)
            
            best_action = None
            min_energy = float('inf')
            
            for action_idx in valid_acts:
                if action_idx < cluster.num_resources:
                    res = c_resources[action_idx]
                    est_time = env_ref.execution_model.compute_execution_time(active_task, res)
                    est_energy = est_time * res.power_rate
                    if est_energy < min_energy:
                        min_energy = est_energy
                        best_action = action_idx
                        
            if best_action is not None:
                actions[aid] = int(best_action)
            else:
                actions[aid] = len(mask) - 2
        return actions
