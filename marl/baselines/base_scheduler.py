import abc
from typing import Dict, Any

class Scheduler(abc.ABC):
    """Abstract base class for scheduling algorithms.
    
    All schedulers (heuristic and learned) share this interface
    so they can be evaluated on the identical environment for
    fair comparison.
    
    Status: Implemented
    """
    def __init__(self, name: str):
        self.name = name
    
    @abc.abstractmethod
    def select_actions(self, env: Any) -> Dict[int, int]:
        """Given current env state, return action for each agent.
        Action = index into agent's action space (resource index or defer)."""
        pass
    
    def reset(self) -> None:
        """Reset any internal state between episodes."""
        pass
