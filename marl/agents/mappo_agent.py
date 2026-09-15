import numpy as np
import torch
from typing import Tuple, Optional, Any

from marl.agents.base_agent import BaseAgent
from marl.agents.actor import ActorNetwork
from marl.agents.critic import CentralizedCritic

class MAPPOAgent(BaseAgent):
    """MAPPO agent with actor (decentralized) and critic (centralized) references.
    
    Status: Implemented
    """
    def __init__(self, agent_id: int, obs_dim: int, action_dim: int,
                 global_state_dim: int, config: dict,
                 actor: Optional[ActorNetwork] = None,
                 critic: Optional[CentralizedCritic] = None):
        super().__init__(agent_id, obs_dim, action_dim, config)
        
        self.global_state_dim = global_state_dim
        self._device = torch.device(config.get('device', 'cpu'))
        
        hidden_sizes = config.get('hidden_sizes', [128, 128])
        activation = config.get('activation', 'relu')
        
        if actor is None:
            self.actor = ActorNetwork(obs_dim, action_dim, hidden_sizes, activation).to(self._device)
        else:
            self.actor = actor
            
        if critic is None:
            self.critic = CentralizedCritic(global_state_dim, [256, 256], activation).to(self._device)
        else:
            self.critic = critic

    @property
    def device(self) -> torch.device:
        return self._device

    def select_action(self, obs: np.ndarray, action_mask: np.ndarray) -> Tuple[int, float, float]:
        """Decentralized action selection.
        Returns (action_idx, log_prob, value_estimate).
        Uses actor for action, critic needs global_state (stored separately)."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        mask_t = torch.as_tensor(action_mask, dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            dist = self.actor(obs_t, mask_t)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
        return int(action.item()), log_prob.item(), 0.0
    
    def get_value(self, global_state: np.ndarray) -> float:
        """Centralized value estimate using the critic."""
        state_t = torch.as_tensor(global_state, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            val = self.critic(state_t)
        return val.item()
    
    def select_action_deterministic(self, obs: np.ndarray, action_mask: np.ndarray) -> int:
        """For evaluation: argmax over valid actions."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        mask_t = torch.as_tensor(action_mask, dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            logits = self.actor.net(obs_t)
            logits = logits.masked_fill(mask_t == 0, -1e8)
            action = torch.argmax(logits, dim=-1)
            
        return int(action.item())
        
    def get_action_distribution(self, obs: np.ndarray, action_mask: np.ndarray):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        mask_t = torch.as_tensor(action_mask, dtype=torch.float32, device=self.device)
        return self.actor(obs_t, mask_t)
