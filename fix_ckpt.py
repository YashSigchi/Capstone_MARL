import torch
import json

# Update json
with open('checkpoints/best_model_metadata.json', 'r') as f:
    data = json.load(f)
    
data['metadata']['validation_completion_rate'] = data['metrics']['task_completion_rate_mean']
data['metadata']['validation_rejection_rate'] = data['metrics']['task_rejection_rate_mean']
data['metadata']['validation_sla_violation_rate'] = data['metrics']['sla_violation_rate_mean']
data['metadata']['validation_latency'] = data['metrics']['avg_task_latency_mean']
data['metadata']['validation_energy_per_completed_task'] = data['metrics']['energy_per_completed_task_mean']

with open('checkpoints/best_model_metadata.json', 'w') as f:
    json.dump(data, f, indent=2)
    
# Update pt
ckpt = torch.load('checkpoints/best_model.pt')
ckpt['metadata']['validation_completion_rate'] = data['metrics']['task_completion_rate_mean']
ckpt['metadata']['validation_rejection_rate'] = data['metrics']['task_rejection_rate_mean']
ckpt['metadata']['validation_sla_violation_rate'] = data['metrics']['sla_violation_rate_mean']
ckpt['metadata']['validation_latency'] = data['metrics']['avg_task_latency_mean']
ckpt['metadata']['validation_energy_per_completed_task'] = data['metrics']['energy_per_completed_task_mean']
torch.save(ckpt, 'checkpoints/best_model.pt')
