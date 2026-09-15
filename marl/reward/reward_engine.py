import json
from dataclasses import dataclass
from typing import Dict, Any

from marl.reward.reward_components import REWARD_COMPONENTS
from marl.reward.reward_normalizer import RewardNormalizer
from marl.reward.dynamic_weight_selector import DynamicWeightSelector

# Status: Implemented

@dataclass
class RewardResult:
    total: float
    local: float
    global_: float
    components: Dict[str, float]
    weighted_components: Dict[str, float]

class RewardEngine:
    """
    Computes blended multi-objective reward.
    R = alpha * R_local + (1-alpha) * R_global
    """
    def __init__(self, config: dict):
        self.config = config
        self.alpha = config.get('reward_blend_alpha', 0.5)
        self.invalid_action_penalty = config.get('invalid_action_penalty', -1.0)
        
        self.normalizer = RewardNormalizer(
            method=config.get('normalization', 'running_mean_std'),
            num_components=len(REWARD_COMPONENTS)
        )
        
        base_weights = config.get('weights', {k: 1.0/len(REWARD_COMPONENTS) for k in REWARD_COMPONENTS})
        constraints = config.get('dynamic_weights_constraints', None)
        mode = config.get('mode', 'fixed')
        
        self.weight_selector = DynamicWeightSelector(mode, base_weights, constraints)

    def compute(self, env_info: Dict[str, Any]) -> RewardResult:
        """Compute the final reward and its components."""
        if env_info.get('invalid_action', False):
            return RewardResult(
                total=self.invalid_action_penalty,
                local=self.invalid_action_penalty,
                global_=0.0,
                components={},
                weighted_components={}
            )
            
        components = {}
        norm_components = {}
        
        # 1. Compute raw components & 2. Normalize
        for name, comp_info in REWARD_COMPONENTS.items():
            val = comp_info['fn'](env_info)
            components[name] = val
            self.normalizer.update(name, val)
            norm_components[name] = self.normalizer.normalize(name, val)
            
        # 3. Get weights
        env_metrics = {
            'system_avg_utilization': env_info.get('system_avg_utilization', 0.0),
            'sla_violation_rate': env_info.get('total_sla_violations', 0) / max(1, env_info.get('total_tasks_processed', 1)),
            'total_energy': env_info.get('total_energy', 0.0)
        }
        weights = self.weight_selector.get_weights(env_metrics)
        
        # 4. Separate local & global
        r_local = 0.0
        r_global = 0.0
        weighted_comps = {}
        
        for name, comp_info in REWARD_COMPONENTS.items():
            w = weights.get(name, 0.0)
            weighted_val = w * norm_components[name]
            weighted_comps[name] = weighted_val
            
            if comp_info['scope'] == 'local':
                r_local += weighted_val
            else:
                r_global += weighted_val
                
        # 7. Total reward
        r_total = self.alpha * r_local + (1.0 - self.alpha) * r_global
        
        return RewardResult(
            total=r_total,
            local=r_local,
            global_=r_global,
            components=components,
            weighted_components=weighted_comps
        )

    def reset(self) -> None:
        """Reset the engine's normalizer statistics."""
        self.normalizer.reset()

    def get_current_weights(self) -> Dict[str, float]:
        """Returns the current weights."""
        return self.weight_selector.get_weights()

    @classmethod
    def from_config(cls, config_path: str) -> 'RewardEngine':
        """Load RewardEngine from a JSON configuration file."""
        with open(config_path, 'r') as f:
            config = json.load(f)
        return cls(config)
