import yaml

with open('configs/environment.yaml', 'r') as f:
    env_cfg = yaml.safe_load(f)

# Sanity Config
env_cfg['task_arrival']['rate'] = 0.5
env_cfg['task_distributions']['cpu_req']['max'] = 8.0
env_cfg['task_distributions']['gpu_req']['max'] = 0.0
env_cfg['max_steps'] = 200
env_cfg['episode_length'] = 200

# Make sure resources have different speeds for baseline diversity
env_cfg['resources'] = [
    {'id': 'R1', 'cpu_capacity': 16, 'memory_capacity': 32, 'gpu_capacity': 1, 'power_rate': 250, 'processing_speed': 1.0, 'network_bandwidth': 10.0},
    {'id': 'R2', 'cpu_capacity': 16, 'memory_capacity': 32, 'gpu_capacity': 1, 'power_rate': 250, 'processing_speed': 2.0, 'network_bandwidth': 10.0},
    {'id': 'R3', 'cpu_capacity': 16, 'memory_capacity': 32, 'gpu_capacity': 1, 'power_rate': 250, 'processing_speed': 1.0, 'network_bandwidth': 10.0},
    {'id': 'R4', 'cpu_capacity': 16, 'memory_capacity': 32, 'gpu_capacity': 1, 'power_rate': 250, 'processing_speed': 2.0, 'network_bandwidth': 10.0},
    {'id': 'R5', 'cpu_capacity': 16, 'memory_capacity': 32, 'gpu_capacity': 1, 'power_rate': 250, 'processing_speed': 1.0, 'network_bandwidth': 10.0},
    {'id': 'R6', 'cpu_capacity': 16, 'memory_capacity': 32, 'gpu_capacity': 1, 'power_rate': 250, 'processing_speed': 2.0, 'network_bandwidth': 10.0}
]

env_cfg['clusters'] = [
    {'id': 'C1', 'resource_ids': ['R1', 'R2']},
    {'id': 'C2', 'resource_ids': ['R3', 'R4']},
    {'id': 'C3', 'resource_ids': ['R5', 'R6']}
]

with open('configs/environment.yaml', 'w') as f:
    yaml.dump(env_cfg, f)
