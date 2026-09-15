from typing import Dict, Any
import numpy as np
from marl.baselines.base_scheduler import Scheduler

class RandomFeasibleScheduler(Scheduler):
    """Random Feasible: randomly selects among valid resources.
    
    Stochastic lower bound for comparison.
    
    Status: Baseline
    """
    def __init__(self, seed: int = 42):
        super().__init__(name='random_feasible')
        self.rng = np.random.default_rng(seed)
        
    def select_actions(self, env: Any) -> Dict[int, int]:
        actions = {}
        for agent_id in env.agent_ids:
            mask = env.get_action_mask(agent_id)
            valid_actions = np.where(mask == 1)[0]
            if len(valid_actions) > 0:
                action = self.rng.choice(valid_actions)
            else:
                action = len(mask) - 2
            actions[agent_id] = int(action)
        return actions
