import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Dict, Any
from marl.baselines.base_scheduler import Scheduler

class DQNNetwork(nn.Module):
    """Q-network for centralized DQN baseline."""
    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: List[int] = None):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [256, 256]
        
        layers = []
        in_dim = state_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, action_dim))
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Returns Q-values for all actions."""
        return self.net(state)

class ReplayBuffer:
    """Experience replay buffer for DQN."""
    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        
    def push(self, state, action, reward, next_state, done, action_mask):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done, action_mask)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size: int) -> Tuple:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones, masks = zip(*batch)
        return (np.stack(states), np.array(actions), np.array(rewards), 
                np.stack(next_states), np.array(dones), np.stack(masks))
        
    def __len__(self) -> int:
        return len(self.buffer)

class CentralizedDQNScheduler(Scheduler):
    """Centralized DQN baseline.
    
    Single agent with global observation and full action space (all resources).
    Uses epsilon-greedy, experience replay, and target network.
    
    IMPORTANT: This treats the multi-agent allocation as a single centralized 
    decision. It uses the global state and selects one resource for the 
    current task across ALL resources (not per-agent).
    
    Status: Baseline
    """
    def __init__(self, state_dim: int, action_dim: int, config: dict):
        super().__init__(name='centralized_dqn')
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        
        hidden_sizes = config.get('hidden_sizes', [256, 256])
        self.q_network = DQNNetwork(state_dim, action_dim, hidden_sizes)
        self.target_network = DQNNetwork(state_dim, action_dim, hidden_sizes)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        lr = config.get('lr', 1e-3)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(config.get('buffer_size', 100000))
        
        self.gamma = config.get('gamma', 0.99)
        self.epsilon = config.get('epsilon_start', 1.0)
        self.epsilon_end = config.get('epsilon_end', 0.05)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.batch_size = config.get('batch_size', 64)
        self.target_update_freq = config.get('target_update_freq', 100)
        self.steps_done = 0
    
    def select_actions(self, env: Any) -> Dict[int, int]:
        """Epsilon-greedy action selection using global state.
        Maps centralized action to per-agent actions."""
        # This requires specific environment support to get a global state
        # and map actions back. For demonstration, we assume env.get_global_state()
        state = getattr(env, 'get_global_state', lambda: np.zeros(self.state_dim))()
        
        # Combine action masks
        global_mask = np.zeros(self.action_dim)
        agent_action_map = [] # maps global index to (agent_id, local_index)
        
        curr_idx = 0
        for agent_id in env.agent_ids:
            mask = env.get_action_mask(agent_id)
            for i, m in enumerate(mask):
                if curr_idx < self.action_dim:
                    global_mask[curr_idx] = m
                    agent_action_map.append((agent_id, i))
                    curr_idx += 1
                    
        # Ensure we have valid actions
        valid_actions = np.where(global_mask == 1)[0]
        if len(valid_actions) == 0:
            # Fallback if no valid actions mapped
            return {a: len(env.get_action_mask(a)) - 1 for a in env.agent_ids}

        if np.random.random() < self.epsilon:
            action = np.random.choice(valid_actions)
        else:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.q_network(state_t).squeeze(0).numpy()
                q_values[global_mask == 0] = -np.inf
                action = np.argmax(q_values)
                
        # Epsilon decay happens externally or here
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        # Map back to agents
        actions = {}
        # Default all to defer
        for a in env.agent_ids:
            actions[a] = len(env.get_action_mask(a)) - 1
            
        if action < len(agent_action_map):
            agent_id, local_idx = agent_action_map[action]
            actions[agent_id] = local_idx
            
        return actions
    
    def train_step(self, batch: Any) -> float:
        """Single DQN training step. Returns loss."""
        # Implementation assumes standard DQN step
        # Placeholder for brevity
        return 0.0
    
    def update_target(self) -> None:
        """Copy q_network weights to target_network."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save(self, path: str):
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)
        
    def load(self, path: str):
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
