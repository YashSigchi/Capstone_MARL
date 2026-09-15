import yaml
with open('configs/environment.yaml', 'r') as f:
    env_cfg = yaml.safe_load(f)

env_cfg['clusters'] = [
    {'id': 'C1', 'resource_ids': ['R1', 'R2']},
    {'id': 'C2', 'resource_ids': ['R3', 'R4']},
    {'id': 'C3', 'resource_ids': ['R5', 'R6']}
]
with open('configs/environment.yaml', 'w') as f:
    yaml.dump(env_cfg, f)
