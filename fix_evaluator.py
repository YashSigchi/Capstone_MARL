import re

with open('marl/evaluation/evaluator.py', 'r') as f:
    content = f.read()

# First, remove the accidentally inserted method
bad_code = """    def _extract_episode_end_metrics(
        self,
        env_infos: Dict[int, Dict[str, Any]]
    ) -> Dict[str, float]:
        \"\"\"Extract episode-level metrics (e.g. utilization, SLA violations).\"\"\"
        metrics = {}
        
        # We can just look at agent 0's info for global stats
        if 0 in env_infos:
            info = env_infos[0]
            metrics['system_utilization'] = info.get('system_avg_utilization', 0.0)
            metrics['total_energy'] = info.get('total_energy', 0.0)
            metrics['total_carbon'] = info.get('total_carbon', 0.0)
            metrics['total_sla_violations'] = float(info.get('total_sla_violations', 0))
            metrics['makespan'] = info.get('makespan', 0.0)
            completed = info.get('total_completed_tasks', 0)
            metrics['completed_tasks'] = float(completed)
            metrics['unfinished_tasks'] = float(info.get('unfinished_tasks', 0.0))
            
            if completed > 0:
                metrics['energy_per_completed_task'] = metrics['total_energy'] / completed
                metrics['carbon_per_completed_task'] = metrics['total_carbon'] / completed
            else:
                metrics['energy_per_completed_task'] = 0.0
                metrics['carbon_per_completed_task'] = 0.0
                
        return metrics"""

content = content.replace(bad_code, "")

# Now fix the real _extract_episode_end_metrics (the one that takes episode_metrics)
# Let's see what it looks like around line 150
# Actually we can just add our new metrics to episode_metrics object directly!
# But wait, EpisodeMetrics dataclass in metrics.py doesn't have these fields!
# We should update metrics.py first to add these fields, then set them here.

with open('marl/evaluation/evaluator.py', 'w') as f:
    f.write(content)

