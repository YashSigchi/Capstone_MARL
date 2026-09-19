import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, List
import numpy as np

from marl.algorithms.base_algorithm import BaseMARLAlgorithm
from marl.agents.base_agent import BaseAgent
from marl.agents.actor import ActorNetwork
from marl.agents.critic import CentralizedCritic
from marl.agents.mappo_agent import MAPPOAgent
from marl.training.rollout_buffer import RolloutBuffer

class MAPPO(BaseMARLAlgorithm):
    """Multi-Agent PPO with Centralized Training, Decentralized Execution.
    
    - Shared or individual actor networks
    - Shared centralized critic
    - PPO clipped objective
    - GAE for advantage estimation
    
    Status: Implemented
    """
    def __init__(self, config: dict):
        self.config = config
        self.num_agents = config.get('num_agents', 1)
        self.obs_dim = config.get('obs_dim', 1)
        self.action_dim = config.get('action_dim', 1)
        self.global_state_dim = config.get('global_state_dim', 1)
        self.parameter_sharing = config.get('parameter_sharing', True)
        
        self.lr_actor = config.get('lr_actor', 3e-4)
        self.lr_critic = config.get('lr_critic', 1e-3)
        self.gamma = config.get('gamma', 0.99)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.clip_range = config.get('clip_range', 0.2)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.value_coef = config.get('value_coef', 0.5)
        self.max_grad_norm = config.get('max_grad_norm', 0.5)
        self.num_epochs = config.get('num_epochs', 4)
        self.num_minibatches = config.get('num_minibatches', 4)
        device_str = config.get('device', 'cpu')
        if device_str == 'auto':
            device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device_str)
        
        self.agents = {}
        self.actors = []
        self.critic = None
        self.actor_optimizers = []
        self.critic_optimizer = None

    def setup(self, env_config: dict, algorithm_config: dict) -> None:
        actor_config = self.config.get('actor', self.config.get('actor_config', {}))
        critic_config = self.config.get('critic', self.config.get('critic_config', {}))
        
        self.critic = CentralizedCritic(
            state_dim=self.global_state_dim,
            hidden_sizes=critic_config.get('hidden_sizes', [256, 256]),
            activation=critic_config.get('activation', 'relu')
        ).to(self.device)
        
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_critic)
        
        if self.parameter_sharing:
            shared_actor = ActorNetwork(
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                hidden_sizes=actor_config.get('hidden_sizes', [128, 128]),
                activation=actor_config.get('activation', 'relu')
            ).to(self.device)
            self.actors.append(shared_actor)
            self.actor_optimizers.append(optim.Adam(shared_actor.parameters(), lr=self.lr_actor))
            
            for i in range(self.num_agents):
                agent_config = {'device': self.device.type, **actor_config}
                self.agents[i] = MAPPOAgent(
                    agent_id=i, obs_dim=self.obs_dim, action_dim=self.action_dim,
                    global_state_dim=self.global_state_dim, config=agent_config,
                    actor=shared_actor, critic=self.critic
                )
        else:
            for i in range(self.num_agents):
                actor = ActorNetwork(
                    obs_dim=self.obs_dim,
                    action_dim=self.action_dim,
                    hidden_sizes=actor_config.get('hidden_sizes', [128, 128]),
                    activation=actor_config.get('activation', 'relu')
                ).to(self.device)
                self.actors.append(actor)
                self.actor_optimizers.append(optim.Adam(actor.parameters(), lr=self.lr_actor))
                
                agent_config = {'device': self.device.type, **actor_config}
                self.agents[i] = MAPPOAgent(
                    agent_id=i, obs_dim=self.obs_dim, action_dim=self.action_dim,
                    global_state_dim=self.global_state_dim, config=agent_config,
                    actor=actor, critic=self.critic
                )

    def get_agents(self) -> Dict[int, BaseAgent]:
        return self.agents

    def update(self, rollout_buffer: RolloutBuffer) -> Dict[str, float]:
        """PPO update using rollout data.
        Returns dict with: policy_loss, value_loss, entropy, approx_kl, clip_fraction
        """
        metrics = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'approx_kl': [],
            'clip_fraction': []
        }
        
        for _ in range(self.num_epochs):
            for batch in rollout_buffer.get_minibatch_iterator(self.num_minibatches, self.device):
                obs = batch['obs']
                global_state = batch['global_state']
                actions = batch['actions']
                old_log_probs = batch['log_probs']
                advantages = batch['advantages']
                returns = batch['returns']
                action_mask = batch['action_masks']
                agent_ids = batch['agent_ids']
                
                # Normalize advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                policy_loss = 0.0
                entropy_loss = 0.0
                kl_divs = []
                clip_fracs = []
                
                if self.parameter_sharing:
                    actor = self.actors[0]
                    log_probs, entropy = actor.get_log_prob_and_entropy(obs, actions, action_mask)
                    ratio = torch.exp(log_probs - old_log_probs)
                    
                    surr1 = ratio * advantages
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * advantages
                    
                    policy_loss = -torch.min(surr1, surr2).mean()
                    entropy_loss = entropy.mean()
                    
                    with torch.no_grad():
                        kl_div = ((torch.exp(old_log_probs) * (old_log_probs - log_probs)).mean()).item()
                        clip_frac = ((ratio - 1.0).abs() > self.clip_range).float().mean().item()
                        kl_divs.append(kl_div)
                        clip_fracs.append(clip_frac)
                else:
                    policy_loss = 0.0
                    entropy_loss = 0.0
                    for i in range(self.num_agents):
                        mask = (agent_ids == i)
                        if not mask.any():
                            continue
                        
                        actor = self.actors[i]
                        agent_obs = obs[mask]
                        agent_actions = actions[mask]
                        agent_action_mask = action_mask[mask]
                        agent_old_log_probs = old_log_probs[mask]
                        agent_advs = advantages[mask]
                        
                        log_probs, entropy = actor.get_log_prob_and_entropy(agent_obs, agent_actions, agent_action_mask)
                        ratio = torch.exp(log_probs - agent_old_log_probs)
                        
                        surr1 = ratio * agent_advs
                        surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * agent_advs
                        
                        policy_loss += -torch.min(surr1, surr2).sum()
                        entropy_loss += entropy.sum()
                        
                        with torch.no_grad():
                            kl_div = ((torch.exp(agent_old_log_probs) * (agent_old_log_probs - log_probs)).mean()).item()
                            clip_frac = ((ratio - 1.0).abs() > self.clip_range).float().mean().item()
                            kl_divs.append(kl_div)
                            clip_fracs.append(clip_frac)
                            
                    policy_loss /= len(obs)
                    entropy_loss /= len(obs)

                # Value loss
                values = self.critic(global_state)
                value_loss = nn.MSELoss()(values, returns)
                
                # Total loss
                total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss
                
                # Optimization step
                if self.parameter_sharing:
                    self.actor_optimizers[0].zero_grad()
                else:
                    for opt in self.actor_optimizers:
                        opt.zero_grad()
                self.critic_optimizer.zero_grad()
                
                total_loss.backward()
                
                if self.parameter_sharing:
                    nn.utils.clip_grad_norm_(self.actors[0].parameters(), self.max_grad_norm)
                else:
                    for actor in self.actors:
                        nn.utils.clip_grad_norm_(actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                
                if self.parameter_sharing:
                    self.actor_optimizers[0].step()
                else:
                    for opt in self.actor_optimizers:
                        opt.step()
                self.critic_optimizer.step()
                
                metrics['policy_loss'].append(policy_loss.item())
                metrics['value_loss'].append(value_loss.item())
                metrics['entropy'].append(entropy_loss.item())
                metrics['approx_kl'].append(np.mean(kl_divs))
                metrics['clip_fraction'].append(np.mean(clip_fracs))
                
        return {k: np.mean(v) for k, v in metrics.items()}

    def save(self, path: str, metadata: dict = None) -> None:
        state = {
            'critic': self.critic.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'actors': [a.state_dict() for a in self.actors],
            'actor_optimizers': [o.state_dict() for o in self.actor_optimizers],
        }
        if hasattr(self, 'normalizer'):
            state['normalizer'] = self.normalizer.state_dict()
        if metadata:
            state['metadata'] = metadata
        torch.save(state, path)

    def load(self, path: str) -> None:
        state = torch.load(path, map_location=self.device)
        self.critic.load_state_dict(state['critic'])
        self.critic_optimizer.load_state_dict(state['critic_optimizer'])
        for a, s in zip(self.actors, state['actors']):
            a.load_state_dict(s)
        for o, s in zip(self.actor_optimizers, state['actor_optimizers']):
            o.load_state_dict(s)
        if 'normalizer' in state and hasattr(self, 'normalizer'):
            self.normalizer.load_state_dict(state['normalizer'])
            
        metadata = state.get('metadata')
        if metadata:
            print(f"\n--- Loading checkpoint from {path} ---")
            print(f"episode = {metadata.get('episode', 'unknown')}")
            print(f"training_seed = {metadata.get('training_seed', 'unknown')}")
            print(f"max_rejection_rate = {metadata.get('max_rejection_rate', 'unknown')}")
            print(f"target_rejection_rate = {metadata.get('target_rejection_rate', 'unknown')}")
            print(f"validation_completion = {metadata.get('validation_completion_rate', 'unknown')}")
            print(f"validation_rejection = {metadata.get('validation_rejection_rate', 'unknown')}")
            print("---------------------------------------\n")
