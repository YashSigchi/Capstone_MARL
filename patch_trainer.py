import re

with open('marl/training/trainer.py', 'r') as f:
    content = f.read()

new_evaluate = """    def evaluate(self, num_episodes: int = 5, base_seed: int = 1001) -> dict:
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
            
            metrics['eval_reward'].append(ep_reward)
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
"""

content = re.sub(r"    def evaluate\(self, num_episodes: int = 10\) -> Dict\[str, float\]:.*?return \{.*?\}", new_evaluate, content, flags=re.DOTALL)

# Add best model tracking to train()
train_replacement = """
        best_score = float('-inf')
        
        for episode in range(self.num_episodes):
"""
content = content.replace("        for episode in range(self.num_episodes):", train_replacement)

# Inside the train loop, replace the periodic eval block
old_eval_block = """            # Periodic evaluation
            if (episode + 1) % self.eval_interval == 0:
                eval_metrics = self.evaluate()
                print(
                    f"  [Eval] Mean Reward: {eval_metrics['eval_reward_mean']:.2f} "
                    f"± {eval_metrics['eval_reward_std']:.2f}"
                )
                if self.logger:
                    self.logger.log_scalars(eval_metrics, step=episode)

            # Periodic checkpointing
            if (episode + 1) % self.save_interval == 0:
                path = os.path.join(self.save_dir, f'model_ep{episode+1}.pt')
                self.algorithm.save(path)"""

new_eval_block = """            # Periodic evaluation and best model selection
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
"""
content = content.replace(old_eval_block, new_eval_block)

# Fix imports in trainer.py
if "import json" not in content:
    content = "import json\n" + content

with open('marl/training/trainer.py', 'w') as f:
    f.write(content)
