from typing import Dict, Optional
import copy

# Status: Experimental

class DynamicWeightSelector:
    """
    Selects reward weights. Modes: 'fixed' or 'dynamic'.
    
    Constraints always enforced:
    1. w_i >= 0 for all i
    2. sum(w_i) = 1
    3. w_i >= min_bound_i (per component)
    4. w_i <= max_bound_i (per component)
    
    Status: Experimental
    """
    def __init__(self, mode: str, base_weights: Dict[str, float],
                 constraints: Optional[Dict[str, Dict[str, float]]] = None):
        self.mode = mode
        self.base_weights = copy.deepcopy(base_weights)
        self.current_weights = copy.deepcopy(base_weights)
        
        self.constraints = constraints or {}
        for k in self.base_weights.keys():
            if k not in self.constraints:
                self.constraints[k] = {'min': 0.0, 'max': 1.0}
                
        # Ensure base weights meet constraints
        self.current_weights = self._apply_constraints(self.current_weights)

    def get_weights(self, env_metrics: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """Get current weights, possibly updating them dynamically."""
        if self.mode == 'fixed' or env_metrics is None:
            return self.current_weights
            
        weights = copy.deepcopy(self.current_weights)
        
        # Experimental dynamic adjustment heuristics
        sys_util = env_metrics.get('system_avg_utilization', 0.0)
        sla_rate = env_metrics.get('sla_violation_rate', 0.0)
        energy = env_metrics.get('total_energy', 0.0)
        
        if sys_util > 0.85:
            weights['utilization'] = weights.get('utilization', 0.0) * 1.1
            weights['throughput'] = weights.get('throughput', 0.0) * 0.9
            
        if sla_rate > 0.1:
            weights['sla'] = weights.get('sla', 0.0) * 1.2
            
        if energy > 1000.0: # threshold example
            weights['energy'] = weights.get('energy', 0.0) * 1.1
            
        self.current_weights = self._apply_constraints(weights)
        return self.current_weights

    def _apply_constraints(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Applies constraints and re-normalizes to sum to 1."""
        constrained_weights = copy.deepcopy(weights)
        
        max_iterations = 10
        for _ in range(max_iterations):
            # 1. Clamp weights to bounds
            for k in constrained_weights:
                min_b = self.constraints[k].get('min', 0.0)
                max_b = self.constraints[k].get('max', 1.0)
                constrained_weights[k] = max(min_b, min(max_b, constrained_weights[k]))
                
            # 2. Re-normalize
            constrained_weights = self._normalize_weights(constrained_weights)
            
            # 3. Check if any weight violates bounds after re-normalization
            violations = False
            for k, w in constrained_weights.items():
                if w < self.constraints[k].get('min', 0.0) - 1e-5 or w > self.constraints[k].get('max', 1.0) + 1e-5:
                    violations = True
                    break
            if not violations:
                break
                
        return constrained_weights

    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Normalizes weights so they sum to 1.0."""
        total = sum(weights.values())
        if total <= 1e-8:
            return {k: 1.0 / len(weights) for k in weights}
        return {k: v / total for k, v in weights.items()}
