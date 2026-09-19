import re

with open('marl/training/trainer.py', 'r') as f:
    content = f.read()

old_meta = """                    'validation_completion_rate': eval_metrics.get('task_completion_rate', 0.0) if 'eval_metrics' in locals() else 0.0,
                    'validation_rejection_rate': eval_metrics.get('rejected_tasks', 0.0) if 'eval_metrics' in locals() else 0.0,
                    'validation_sla_violation_rate': eval_metrics.get('sla_violation_rate', 0.0) if 'eval_metrics' in locals() else 0.0,
                    'validation_latency': eval_metrics.get('latency', 0.0) if 'eval_metrics' in locals() else 0.0,
                    'validation_energy_per_completed_task': eval_metrics.get('energy_per_completed_task', 0.0) if 'eval_metrics' in locals() else 0.0,"""

new_meta = """                    'validation_completion_rate': eval_metrics.get('task_completion_rate_mean', 0.0) if 'eval_metrics' in locals() else 0.0,
                    'validation_rejection_rate': eval_metrics.get('task_rejection_rate_mean', 0.0) if 'eval_metrics' in locals() else 0.0,
                    'validation_sla_violation_rate': eval_metrics.get('sla_violation_rate_mean', 0.0) if 'eval_metrics' in locals() else 0.0,
                    'validation_latency': eval_metrics.get('avg_task_latency_mean', 0.0) if 'eval_metrics' in locals() else 0.0,
                    'validation_energy_per_completed_task': eval_metrics.get('energy_per_completed_task_mean', 0.0) if 'eval_metrics' in locals() else 0.0,"""

content = content.replace(old_meta, new_meta)

with open('marl/training/trainer.py', 'w') as f:
    f.write(content)
