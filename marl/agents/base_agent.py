import abc
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Any, Dict, List
import numpy as np
import torch

class BaseAgent(ABC):
    """Abstract base class for MARL agents.
    
    Each agent manages a cluster of heterogeneous resources.
    At each step, the environment presents a task, and the agent
    chooses an eligible resource or defers/rejects.
    
    Status: Implemented
    """
    def __init__(self, agent_id: int, obs_dim: int, action_dim: int, config: dict):
        self.agent_id = agent_id
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.config = config
    
    @abstractmethod
    def select_action(self, obs: np.ndarray, action_mask: np.ndarray) -> Tuple[int, float, float]:
        """Returns (action, log_prob, value_estimate)"""
        pass
    
    @abstractmethod
    def get_value(self, global_state: np.ndarray) -> float:
        """Returns the value estimate of the global state."""
        pass
    
    @abstractmethod
    def get_action_distribution(self, obs: np.ndarray, action_mask: np.ndarray):
        """Returns the action distribution given an observation and action mask."""
        pass
    
    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Returns the device the agent is on."""
        pass
