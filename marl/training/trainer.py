"""Main training loop orchestrator.

Handles: environment interaction, rollout collection,
policy updates, logging, checkpointing, evaluation.

Status: Implemented
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Any
import os

from marl.environment.marl_env import MARLResourceAllocationEnv
from marl.reward.reward_engine import RewardEngine
from marl.algorithms.base_algorithm import BaseMARLAlgorithm
from marl.training.rollout_buffer import RolloutBuffer


class Trainer:
    """Main training loop orchestrator.

    Handles: environment interaction, rollout collection,
    policy updates, logging, checkpointing, evaluation.

    Status: Implemented
    """

    def __init__(
        self,
        env: MARLResourceAllocationEnv,
        algorithm: BaseMARLAlgorithm,
        reward_engine: Optional[RewardEngine] = None,
        config: Optional[dict] = None,
        logger: Any = None,
    ):
        self.env = env
        self.algorithm = algorithm
        self.reward_engine = reward_engine
        self.config = config or {}
        self.logger = logger

        self.num_episodes = self.config.get('num_episodes', 1000)
        self.steps_per_episode = self.config.get('steps_per_episode', 200)
        self.eval_interval = self.config.get('eval_interval', 50)
        self.save_interval = self.config.get('save_interval', 100)
        self.save_dir = self.config.get('checkpoint_dir',
                                        self.config.get('save_dir', './checkpoints'))
        os.makedirs(self.save_dir, exist_ok=True)

        # Get dimensions from env methods
        self.num_agents = self.env.num_agents
        self.agent_ids = self.env.agent_ids
        self.obs_dim = self.env.get_obs_dim()
        self.action_dim = self.env.get_action_dim(self.agent_ids[0])
        self.global_state_dim = self.env.get_global_state_dim()

        self.buffer = RolloutBuffer(
            num_agents=self.num_agents,
            num_steps=self.steps_per_episode,
            obs_dim=self.obs_dim,
            global_state_dim=self.global_state_dim,
            action_dim=self.action_dim,
        )

    def train(self) -> Dict[str, List[float]]:
        """Main training loop.

        Returns:
            Training history with episode rewards and losses.
        """
        history: Dict[str, List[float]] = {
            'episode_reward': [],
            'policy_loss': [],
            'value_loss': [],
        }

        agents = self.algorithm.get_agents()


        best_score = float('-inf')
        
        for episode in range(self.num_episodes):

            # Reset returns (obs_dict, infos)
            obs_dict, infos = self.env.reset()
            self.buffer.reset()
            episode_reward = 0.0

            for step in range(self.steps_per_episode):
                global_state = self.env.get_global_state()

                actions: Dict[int, int] = {}
                log_probs: Dict[int, float] = {}
                values: Dict[int, float] = {}

                # Agents select actions
                for aid in self.agent_ids:
                    obs_i = obs_dict[aid]
                    mask_i = self.env.get_action_mask(aid)

                    action, log_prob, _ = agents[aid].select_action(obs_i, mask_i)
                    value = agents[aid].get_value(global_state)

                    actions[aid] = action
                    log_probs[aid] = log_prob
                    values[aid] = value

                # Environment step
                next_obs_dict, env_rewards, terminated, truncated, infos = self.env.step(actions)

                # Compute rewards via reward engine
                final_rewards: Dict[int, float] = {}
                if self.reward_engine is not None:
                    for aid in self.agent_ids:
                        reward_info = infos.get(aid, {}).get('reward_info', {})
                        result = self.reward_engine.compute(reward_info)
                        final_rewards[aid] = result.total
                else:
                    final_rewards = env_rewards

                # Store transitions
                for idx, aid in enumerate(self.agent_ids):
                    episode_reward += final_rewards.get(aid, 0.0)
                    mask_i = self.env.get_action_mask(aid)

                    self.buffer.insert(
                        step=step,
                        agent_id=idx,
                        obs=obs_dict[aid],
                        global_state=global_state,
                        action=actions[aid],
                        log_prob=log_probs[aid],
                        reward=final_rewards.get(aid, 0.0),
                        value=values[aid],
                        terminated=terminated.get(aid, False),
                        truncated=truncated.get(aid, False),
                        action_mask=mask_i,
                    )

                obs_dict = next_obs_dict
                if all(terminated.values()) or all(truncated.values()):
                    break

            # Compute returns and advantages
            last_values: Dict[int, float] = {}
            global_state = self.env.get_global_state()
            for idx, aid in enumerate(self.agent_ids):
                last_values[idx] = agents[aid].get_value(global_state)

            self.buffer.compute_returns_and_advantages(
                gamma=getattr(self.algorithm, 'gamma', 0.99),
                gae_lambda=getattr(self.algorithm, 'gae_lambda', 0.95),
                last_values=last_values,
            )

            # Policy update
            metrics = self.algorithm.update(self.buffer)

            history['episode_reward'].append(episode_reward)
            history['policy_loss'].append(metrics.get('policy_loss', 0.0))
            history['value_loss'].append(metrics.get('value_loss', 0.0))

            # Logging
            if self.logger:
                log_data = {**metrics, 'episode_reward': episode_reward}
                self.logger.log_scalars(log_data, step=episode)

            # Print progress
            if (episode + 1) % 10 == 0 or episode == 0:
                print(
                    f"Episode {episode+1}/{self.num_episodes}  "
                    f"Reward: {episode_reward:.2f}  "
                    f"Policy Loss: {metrics.get('policy_loss', 0):.4f}  "
                    f"Value Loss: {metrics.get('value_loss', 0):.4f}"
                )

            # Periodic evaluation and best model selection
            if (episode + 1) % self.eval_interval == 0:
                eval_metrics = self.evaluate()
                
                # Model selection score rule
                score = (
                    eval_metrics['task_completion_rate'] * 1000.0 
                    - eval_metrics['sla_violation_rate'] * 100.0
                    - eval_metrics['energy_per_completed_task'] * 0.1
                )
                
                print(f"  [Eval] Reward: {eval_metrics['eval_reward']:.2f} | "
                      f"Comp Rate: {eval_metrics['task_completion_rate']*100:.1f}% | "
                      f"Rej: {eval_metrics['rejected_tasks']:.0f} | "
                      f"Score: {score:.2f}")
                
                if score > best_score:
                    best_score = score
                    best_path = os.path.join(self.save_dir, 'best_model.pt')
                    self.algorithm.save(best_path)
                    
                    import json
                    meta_path = os.path.join(self.save_dir, 'best_model_metadata.json')
                    with open(meta_path, 'w') as mf:
                        json.dump({
                            'training_episode': episode + 1,
                            'validation_seeds': [1001 + i for i in range(5)],
                            'selection_score': score,
                            'metrics': eval_metrics
                        }, mf, indent=2)
                        
                if self.logger:
                    self.logger.log_scalars(eval_metrics, step=episode)

            # Periodic checkpointing
            if (episode + 1) % self.save_interval == 0:
                path = os.path.join(self.save_dir, f'model_ep{episode+1}.pt')
                self.algorithm.save(path)
                
        # Final state save
        latest_path = os.path.join(self.save_dir, 'latest_model.pt')
        self.algorithm.save(latest_path)


        return history

    def evaluate(self, num_episodes: int = 5, base_seed: int = 1001) -> dict:
        """Run evaluation on fixed validation seeds."""
        agents = self.algorithm.get_agents()
        
        metrics = {
            'eval_reward': [],
            'task_completion_rate': [],
            'sla_violation_rate': [],
            'latency': [],
            'energy_per_completed_task': [],
            'rejected_tasks': [],
            'completed_tasks': [],
            'unfinished_tasks': []
        }
        
        # Disable reward normalizer update during eval if available
        if self.reward_engine:
            old_method = self.reward_engine.normalizer.method
            self.reward_engine.normalizer.method = 'none'

        for ep in range(num_episodes):
            eval_seed = base_seed + ep
            obs_dict, _ = self.env.reset(seed=eval_seed)
            ep_reward = 0.0
            
            for _ in range(self.steps_per_episode):
                actions = {}
                for aid in self.agent_ids:
                    obs_i = obs_dict[aid]
                    mask_i = self.env.get_action_mask(aid)
                    actions[aid] = agents[aid].select_action_deterministic(obs_i, mask_i)

                next_obs_dict, env_rewards, terminated, truncated, infos = self.env.step(actions)
                
                if self.reward_engine:
                    for aid in self.agent_ids:
                        reward_info = infos.get(aid, {}).get('reward_info', {})
                        result = self.reward_engine.compute(reward_info)
                        ep_reward += result.total
                else:
                    ep_reward += sum(env_rewards.values())
                    
                obs_dict = next_obs_dict
                if all(terminated.values()) or all(truncated.values()):
                    break
            
            report_div = len(self.agent_ids) * self.steps_per_episode
            metrics['eval_reward'].append(ep_reward / max(report_div, 1))
            # Extract env metrics from any agent info
            info = infos.get(self.agent_ids[0], {}).get('reward_info', {})
            total_processed = info.get('total_tasks_processed', 1)
            completed = len(self.env._completed_tasks)
            rejected = len(self.env._rejected_tasks)
            unfinished = sum(q.length for q in self.env.queues.values()) + len(self.env._running_tasks)
            
            metrics['task_completion_rate'].append(completed / max(1, total_processed + unfinished))
            metrics['sla_violation_rate'].append(info.get('total_sla_violations', 0) / max(1, completed))
            metrics['latency'].append(info.get('total_completion_time', 0.0))
            metrics['energy_per_completed_task'].append(info.get('total_energy', 0.0) / max(1, completed))
            metrics['rejected_tasks'].append(rejected)
            metrics['completed_tasks'].append(completed)
            metrics['unfinished_tasks'].append(unfinished)
            
        if self.reward_engine:
            self.reward_engine.normalizer.method = old_method

        import numpy as np
        return {k: float(np.mean(v)) for k, v in metrics.items()}

