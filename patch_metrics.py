import re

with open('marl/evaluation/metrics.py', 'r') as f:
    content = f.read()

# Add fields to EpisodeMetrics
new_fields = """
    # New metrics
    makespan: float = 0.0
    completed_tasks: float = 0.0
    unfinished_tasks: float = 0.0
    energy_per_completed_task: float = 0.0
    carbon_per_completed_task: float = 0.0
"""

content = content.replace("    total_carbon: float = 0.0", "    total_carbon: float = 0.0" + new_fields)

with open('marl/evaluation/metrics.py', 'w') as f:
    f.write(content)

with open('marl/evaluation/evaluator.py', 'r') as f:
    content = f.read()

new_metrics_code = """
        # --- SLA ---
        sla_tracker = getattr(env, 'sla_tracker', None)
        if sla_tracker is not None:
            records = getattr(sla_tracker, 'records', [])
            if records:
                n_violations = sum(1 for r in records if r.violated)
                episode_metrics.sla_violation_rate = n_violations / max(len(records), 1)
                high_prio_violations = sum(
                    1 for r in records
                    if r.violated and getattr(r, 'priority', 0) >= 4
                )
                episode_metrics.high_priority_sla_violations = float(high_prio_violations)
                slacks = [r.slack for r in records if hasattr(r, 'slack')]
                if slacks:
                    episode_metrics.avg_deadline_slack = float(np.mean(slacks))
                    
        # --- New metrics ---
        episode_metrics.makespan = float(getattr(env, '_makespan', 0.0))
        episode_metrics.completed_tasks = float(n_completed)
        episode_metrics.unfinished_tasks = float(sum(q.length for q in env.queues.values()) + len(getattr(env, '_running_tasks', [])))
        if n_completed > 0:
            episode_metrics.energy_per_completed_task = episode_metrics.total_energy / n_completed
            episode_metrics.carbon_per_completed_task = episode_metrics.total_carbon / n_completed
        else:
            episode_metrics.energy_per_completed_task = 0.0
            episode_metrics.carbon_per_completed_task = 0.0
"""

content = content.replace("        # --- Energy & carbon ---\n        episode_metrics.total_energy = float(getattr(env, '_total_energy', 0.0))\n        episode_metrics.total_carbon = float(getattr(env, '_total_carbon', 0.0))", "        # --- Energy & carbon ---\n        episode_metrics.total_energy = float(getattr(env, '_total_energy', 0.0))\n        episode_metrics.total_carbon = float(getattr(env, '_total_carbon', 0.0))" + new_metrics_code)

with open('marl/evaluation/evaluator.py', 'w') as f:
    f.write(content)
