import numpy as np
import torch
from typing import Dict, List, Iterator

class RolloutBuffer:
    """Stores rollout data for PPO-style on-policy training.
    
    Stores per-agent per-step: obs, global_state, action, log_prob,
    reward, value, terminated, truncated, action_mask.
    
    Status: Implemented
    """
    def __init__(self, num_agents: int, num_steps: int, obs_dim: int,
                 global_state_dim: int, action_dim: int):
        self.num_agents = num_agents
        self.num_steps = num_steps
        self.obs_dim = obs_dim
        self.global_state_dim = global_state_dim
        self.action_dim = action_dim
        
        self.obs = np.zeros((num_steps, num_agents, obs_dim), dtype=np.float32)
        self.global_states = np.zeros((num_steps, global_state_dim), dtype=np.float32)
        self.actions = np.zeros((num_steps, num_agents), dtype=np.float32)
        self.log_probs = np.zeros((num_steps, num_agents), dtype=np.float32)
        self.rewards = np.zeros((num_steps, num_agents), dtype=np.float32)
        self.values = np.zeros((num_steps, num_agents), dtype=np.float32)
        self.terminated = np.zeros((num_steps, num_agents), dtype=np.float32)
        self.truncated = np.zeros((num_steps, num_agents), dtype=np.float32)
        self.action_masks = np.zeros((num_steps, num_agents, action_dim), dtype=np.float32)
        
        self.returns = np.zeros((num_steps, num_agents), dtype=np.float32)
        self.advantages = np.zeros((num_steps, num_agents), dtype=np.float32)
        
        self.step = 0

    def insert(self, step: int, agent_id: int, obs: np.ndarray, global_state: np.ndarray, action: int,
               log_prob: float, reward: float, value: float, terminated: bool, truncated: bool, action_mask: np.ndarray):
        self.obs[step, agent_id] = obs
        if agent_id == 0:
            self.global_states[step] = global_state
        self.actions[step, agent_id] = action
        self.log_probs[step, agent_id] = log_prob
        self.rewards[step, agent_id] = reward
        self.values[step, agent_id] = value
        self.terminated[step, agent_id] = float(terminated)
        self.truncated[step, agent_id] = float(truncated)
        self.action_masks[step, agent_id] = action_mask
        self.step = step + 1

    def compute_returns_and_advantages(self, gamma: float, gae_lambda: float,
                                        last_values: Dict[int, float]) -> None:
        """GAE computation.
        delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
        A_t = sum_{l=0}^{T-t} (gamma * lambda)^l * delta_{t+l}
        returns = advantages + values
        """
        last_gae_lam = np.zeros(self.num_agents, dtype=np.float32)
        
        for step in reversed(range(self.num_steps)):
            if step == self.num_steps - 1:
                next_non_terminal = 1.0 - self.terminated[step]
                next_values = np.array([last_values.get(i, 0.0) for i in range(self.num_agents)])
            else:
                next_non_terminal = 1.0 - self.terminated[step]
                next_values = self.values[step + 1]
                
            delta = self.rewards[step] + gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
            
        self.returns = self.advantages + self.values

    def get_minibatch_iterator(self, num_minibatches: int, device: torch.device) -> Iterator:
        """Yields minibatches as dicts of tensors. Shuffles indices."""
        batch_size = self.num_steps * self.num_agents
        minibatch_size = batch_size // num_minibatches
        
        indices = np.random.permutation(batch_size)
        
        obs_flat = self.obs.reshape(-1, self.obs_dim)
        global_states_flat = np.repeat(self.global_states, self.num_agents, axis=0)
        actions_flat = self.actions.reshape(-1)
        log_probs_flat = self.log_probs.reshape(-1)
        advantages_flat = self.advantages.reshape(-1)
        returns_flat = self.returns.reshape(-1)
        action_masks_flat = self.action_masks.reshape(-1, self.action_dim)
        agent_ids_flat = np.tile(np.arange(self.num_agents), self.num_steps)
        
        for start_idx in range(0, batch_size, minibatch_size):
            end_idx = min(start_idx + minibatch_size, batch_size)
            mb_indices = indices[start_idx:end_idx]
            
            yield {
                'obs': torch.as_tensor(obs_flat[mb_indices], dtype=torch.float32, device=device),
                'global_state': torch.as_tensor(global_states_flat[mb_indices], dtype=torch.float32, device=device),
                'actions': torch.as_tensor(actions_flat[mb_indices], dtype=torch.float32, device=device),
                'log_probs': torch.as_tensor(log_probs_flat[mb_indices], dtype=torch.float32, device=device),
                'advantages': torch.as_tensor(advantages_flat[mb_indices], dtype=torch.float32, device=device),
                'returns': torch.as_tensor(returns_flat[mb_indices], dtype=torch.float32, device=device),
                'action_masks': torch.as_tensor(action_masks_flat[mb_indices], dtype=torch.float32, device=device),
                'agent_ids': torch.as_tensor(agent_ids_flat[mb_indices], dtype=torch.long, device=device)
            }

    def reset(self) -> None:
        self.step = 0
