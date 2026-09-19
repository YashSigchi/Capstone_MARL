from typing import Dict, Any, List
import numpy as np
from marl.baselines.base_scheduler import Scheduler
from marl.environment.marl_env import MARLResourceAllocationEnv

class GreedySLAScheduler(Scheduler):
    """
    Allocates tasks to the feasible resource that minimizes completion time risk 
    (maximizes slack = deadline - estimated_completion_time).
    """
    def __init__(self):
        super().__init__(name='greedy_sla')

    def select_actions(self, env_ref: MARLResourceAllocationEnv):
        actions = {}
        current_time = env_ref.current_step
        
        for aid in env_ref.agent_ids:
            mask = env_ref.get_action_mask(aid)
            valid_acts = np.where(mask == 1.0)[0]
            
            if len(valid_acts) == 0:
                actions[aid] = len(mask) - 2
                continue
                
            active_task = env_ref.queues[aid].peek()
            if active_task is None:
                actions[aid] = len(mask) - 2
                continue
                
            cluster = env_ref._get_cluster(aid)
            c_resources = cluster.get_resources(env_ref.resources)
            
            best_action = None
            max_slack = -float('inf')
            
            for action_idx in valid_acts:
                if action_idx < cluster.num_resources:
                    res = c_resources[action_idx]
                    est_exec = env_ref.execution_model.compute_execution_time(active_task, res)
                    # Rough completion time: current + exec (ignoring network)
                    est_comp = current_time + est_exec
                    slack = active_task.deadline - est_comp
                    
                    if slack > max_slack:
                        max_slack = slack
                        best_action = action_idx
                        
            if best_action is not None:
                actions[aid] = int(best_action)
            else:
                actions[aid] = len(mask) - 2
        return actions
