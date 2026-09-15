import numpy as np
from typing import Dict

# Status: Implemented

class RewardNormalizer:
    """
    Normalizes reward components to standard ranges or distributions.
    """
    def __init__(self, method: str = 'running_mean_std', num_components: int = 7):
        """
        Args:
            method: 'none', 'min_max', 'running_mean_std', 'fixed_range'
            num_components: Expected number of components (unused strictly, tracked by dicts)
        """
        self.method = method
        self.num_components = num_components
        
        # Welford's online algorithm states
        self.counts: Dict[str, int] = {}
        self.means: Dict[str, float] = {}
        self.M2s: Dict[str, float] = {}
        
        # Min-Max states
        self.mins: Dict[str, float] = {}
        self.maxs: Dict[str, float] = {}

    def normalize(self, component_name: str, value: float) -> float:
        """Normalize the given value according to the chosen method."""
        if not np.isfinite(value):
            return 0.0

        if self.method == 'none':
            return value
            
        elif self.method == 'running_mean_std':
            if component_name not in self.counts or self.counts[component_name] < 2:
                return value
            mean = self.means[component_name]
            var = self.M2s[component_name] / (self.counts[component_name] - 1)
            std = (var ** 0.5) + 1e-8
            result = (value - mean) / std
            return float(np.clip(result, -10.0, 10.0))
            
        elif self.method == 'min_max':
            if component_name not in self.mins:
                return value
            min_v = self.mins[component_name]
            max_v = self.maxs[component_name]
            if max_v - min_v < 1e-8:
                return 0.0
            return (value - min_v) / (max_v - min_v)
            
        elif self.method == 'fixed_range':
            # Rely on components being roughly in [-1, 1] or [0, 1] bounds already.
            return value
            
        return value

    def update(self, component_name: str, value: float) -> None:
        """Update statistics with a new value."""
        if not np.isfinite(value):
            return

        if component_name not in self.counts:
            self.counts[component_name] = 0
            self.means[component_name] = 0.0
            self.M2s[component_name] = 0.0
            self.mins[component_name] = float('inf')
            self.maxs[component_name] = float('-inf')
            
        self.counts[component_name] += 1
        delta = value - self.means[component_name]
        self.means[component_name] += delta / self.counts[component_name]
        delta2 = value - self.means[component_name]
        self.M2s[component_name] += delta * delta2
        
        if value < self.mins[component_name]:
            self.mins[component_name] = value
        if value > self.maxs[component_name]:
            self.maxs[component_name] = value

    def reset(self) -> None:
        """Reset all normalization statistics."""
        self.counts.clear()
        self.means.clear()
        self.M2s.clear()
        self.mins.clear()
        self.maxs.clear()
