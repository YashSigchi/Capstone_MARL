import torch
import torch.nn as nn
from typing import List
import numpy as np

class CentralizedCritic(nn.Module):
    """Centralized value function for CTDE.
    
    Input: global state (concatenation of all observations + global features)
    Output: scalar value estimate V(s)
    
    Shared across all agents during training.
    
    Status: Implemented
    """
    def __init__(self, state_dim: int, hidden_sizes: List[int] = [256, 256],
                 activation: str = 'relu'):
        super().__init__()
        
        act_dict = {
            'relu': nn.ReLU,
            'tanh': nn.Tanh,
            'elu': nn.ELU
        }
        act_fn = act_dict.get(activation.lower(), nn.ReLU)
        
        layers = []
        in_dim = state_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(act_fn())
            in_dim = h
            
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
        
        # Orthogonal initialization
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
                
        # Output layer specific init
        nn.init.orthogonal_(self.net[-1].weight, gain=1.0)
    
    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        """Returns value estimate. Shape: (batch,) or (batch, 1) squeezed."""
        v = self.net(global_state)
        return v.squeeze(-1)
