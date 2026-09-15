from typing import Dict, Any
from marl.baselines.base_scheduler import Scheduler

class RoundRobinScheduler(Scheduler):
    """Round-Robin: cycles through resources in order.
    
    Maintains a per-agent counter that advances each step.
    If the next resource in rotation is infeasible, skip to next valid one.
    
    Status: Baseline
    """
    def __init__(self):
        super().__init__(name='round_robin')
        self._counters: Dict[int, int] = {}  # agent_id -> current resource index
    
    def select_actions(self, env: Any) -> Dict[int, int]:
        actions = {}
        for agent_id in env.agent_ids:
            if agent_id not in self._counters:
                self._counters[agent_id] = 0
                
            mask = env.get_action_mask(agent_id)
            num_resources = len(mask) - 2
            start_idx = self._counters[agent_id]
            
            action = len(mask) - 2 # Default to defer
            if num_resources > 0:
                for _ in range(num_resources):
                    idx = self._counters[agent_id] % num_resources
                    self._counters[agent_id] += 1
                    if mask[idx] == 1:
                        action = idx
                        break
            
            actions[agent_id] = action
        return actions
    
    def reset(self) -> None:
        self._counters.clear()
