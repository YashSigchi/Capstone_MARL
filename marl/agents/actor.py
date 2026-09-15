import torch
import torch.nn as nn
from torch.distributions import Categorical
from typing import List, Tuple, Optional
import numpy as np

class ActorNetwork(nn.Module):
    """Policy network mapping observations to action distributions.
    
    Applies action masking: sets logits for invalid actions to -inf
    before computing softmax/Categorical distribution.
    
    Supports parameter sharing (same network for all agents) or
    individual networks.
    
    Status: Implemented
    """
    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: List[int] = [128, 128],
                 activation: str = 'relu'):
        super().__init__()
        
        act_dict = {
            'relu': nn.ReLU,
            'tanh': nn.Tanh,
            'elu': nn.ELU
        }
        act_fn = act_dict.get(activation.lower(), nn.ReLU)
        
        layers = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(act_fn())
            in_dim = h
            
        layers.append(nn.Linear(in_dim, action_dim))
        self.net = nn.Sequential(*layers)
        
        # Orthogonal initialization
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
        
        # Output layer specific init
        nn.init.orthogonal_(self.net[-1].weight, gain=0.01)

    def forward(self, obs: torch.Tensor, action_mask: Optional[torch.Tensor] = None) -> Categorical:
        """Returns Categorical distribution over actions.
        If action_mask provided (1=valid, 0=invalid), set invalid logits to -1e8."""
        logits = self.net(obs)
        if action_mask is not None:
            # action_mask is 1 for valid, 0 for invalid
            logits = logits.masked_fill(action_mask == 0, -1e8)
        
        return Categorical(logits=logits)
    
    def get_log_prob_and_entropy(self, obs: torch.Tensor, actions: torch.Tensor,
                                  action_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """For PPO update: returns log_probs and entropy for given obs-action pairs."""
        dist = self.forward(obs, action_mask)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy
