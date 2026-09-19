import re

with open('marl/training/trainer.py', 'r') as f:
    content = f.read()

init_patch_target = "        self.eval_interval = self.config.get('eval_interval', 50)"
init_patch_replacement = """        self.eval_interval = self.config.get('eval_interval', 50)
        self.eval_episodes = self.config.get('eval_episodes', 10)
        
        import collections
        self.recent_scores = collections.deque(maxlen=3)"""
content = content.replace(init_patch_target, init_patch_replacement)

eval_patch_target = "eval_metrics = self.evaluate()"
eval_patch_replacement = "eval_metrics = self.evaluate(num_episodes=self.eval_episodes)"
content = content.replace(eval_patch_target, eval_patch_replacement)

# Update smoothing logic
old_score_logic = """                score = (
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

new_score_logic = """                raw_score = (
                    round(eval_metrics.get('task_completion_rate_mean', 0.0), 3),
                    -round(eval_metrics.get('task_rejection_rate_mean', 0.0), 3),
                    -round(eval_metrics.get('sla_violation_rate_mean', 0.0), 3),
                    round(eval_metrics.get('throughput_mean', 0.0), 3),
                    -round(eval_metrics.get('avg_task_latency_mean', 0.0), 1),
                    -round(eval_metrics.get('energy_per_completed_task_mean', 0.0), 1)
                )
                
                self.recent_scores.append(raw_score)
                # Compute smoothed score
                smoothed_score = tuple(
                    round(sum(col) / len(self.recent_scores), 3) 
                    for col in zip(*self.recent_scores)
                )
                
                score = smoothed_score
                
                print(f"Eval Ep {episode+1} | "
                      f"Comp Rate: {eval_metrics.get('task_completion_rate_mean', 0.0)*100:.1f}% | "
                      f"Rej: {eval_metrics.get('task_rejection_rate_mean', 0.0):.3f} | "
                      f"Raw Score: {raw_score} | Smoothed: {smoothed_score}")"""

content = content.replace(old_score_logic, new_score_logic)

with open('marl/training/trainer.py', 'w') as f:
    f.write(content)
