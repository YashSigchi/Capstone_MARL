from abc import ABC, abstractmethod
from typing import Dict, Any
from marl.agents.base_agent import BaseAgent

class BaseMARLAlgorithm(ABC):
    """Abstract base for MARL algorithms.
    
    Designed so that MAPPO can be replaced by QMIX, MADDPG, etc.
    without changing the environment or training infrastructure.
    """
    @abstractmethod
    def setup(self, env_config: dict, algorithm_config: dict) -> None:
        pass
    
    @abstractmethod
    def get_agents(self) -> Dict[int, BaseAgent]:
        pass
    
    @abstractmethod
    def update(self, rollout_data: Any) -> Dict[str, float]:
        """Perform policy update. Returns loss metrics dict."""
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        pass
