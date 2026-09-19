import re

with open('marl/training/trainer.py', 'r') as f:
    content = f.read()

old_eval = """    def evaluate(self, num_episodes: int = 5, base_seed: int = 1001) -> dict:
        \"\"\"Run evaluation on fixed validation seeds.\"\"\"
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
        return {k: float(np.mean(v)) for k, v in metrics.items()}"""

new_eval = """    def evaluate(self, num_episodes: int = 5, base_seed: int = 1001) -> dict:
        \"\"\"Run evaluation on fixed validation seeds.\"\"\"
        agents = self.algorithm.get_agents()
        if self.reward_engine:
            old_method = self.reward_engine.normalizer.method
            self.reward_engine.normalizer.method = 'none'
            
        old_seed = getattr(self.evaluator, 'seed', base_seed)
        self.evaluator.seed = base_seed
        
        # We need to temporarily replace the evaluator's reset to use eval seeds.
        # But wait, self.evaluator._run_episode does: ep_seed = self.seed + self._episode_run_counter
        # So we just set self.evaluator.seed = base_seed and it works!
        summary = self.evaluator.evaluate_agents(agents, num_episodes=num_episodes, deterministic=True)
        
        self.evaluator.seed = old_seed
        if self.reward_engine:
            self.reward_engine.normalizer.method = old_method
            
        # Map metrics to expected keys
        return summary"""

content = content.replace(old_eval, new_eval)

# Replace scoring logic
old_score = """                score = (
                    round(eval_metrics['task_completion_rate'], 3),
                    -round(eval_metrics.get('task_rejection_rate', eval_metrics.get('rejected_tasks', 0) / max(1, eval_metrics.get('total_tasks', 100))), 3),
                    -round(eval_metrics.get('sla_violation_rate', 0.0), 3),
                    round(eval_metrics.get('throughput', 0.0), 3),
                    -round(eval_metrics.get('avg_task_latency', 0.0), 1),
                    -round(eval_metrics.get('energy_per_completed_task', 0.0), 1)
                )
                
                print(f"Eval Ep {episode+1} | "
                      f"Comp Rate: {eval_metrics['task_completion_rate']*100:.1f}% | "
                      f"Rej: {eval_metrics['rejected_tasks']:.0f} | "
                      f"Score Tuple: {score}")"""

new_score = """                score = (
                    round(eval_metrics.get('task_completion_rate_mean', 0.0), 3),
                    -round(eval_metrics.get('task_rejection_rate_mean', 0.0), 3),
                    -round(eval_metrics.get('sla_violation_rate_mean', 0.0), 3),
                    round(eval_metrics.get('throughput_mean', 0.0), 3),
                    -round(eval_metrics.get('avg_task_latency_mean', 0.0), 1),
                    -round(eval_metrics.get('energy_per_completed_task_mean', 0.0), 1)
                )
                
                print(f"Eval Ep {episode+1} | "
                      f"Comp Rate: {eval_metrics.get('task_completion_rate_mean', 0.0)*100:.1f}% | "
                      f"Rej: {eval_metrics.get('task_rejection_rate_mean', 0.0):.3f} | "
                      f"Score Tuple: {score}")"""

content = content.replace(old_score, new_score)

with open('marl/training/trainer.py', 'w') as f:
    f.write(content)
