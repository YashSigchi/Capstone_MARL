import yaml
with open('configs/environment.yaml', 'r') as f:
    env_cfg = yaml.safe_load(f)

# Keep the resources but change agent_mapping
with open('configs/marl.yaml', 'r') as f:
    marl_cfg = yaml.safe_load(f)

marl_cfg['agent_mapping'] = {'0': 'C1', '1': 'C2', '2': 'C3'}
with open('configs/marl.yaml', 'w') as f:
    yaml.dump(marl_cfg, f)

env_cfg['clusters'] = [
    {'cluster_id': 'C1', 'resource_ids': ['R1', 'R2']},
    {'cluster_id': 'C2', 'resource_ids': ['R3', 'R4']},
    {'cluster_id': 'C3', 'resource_ids': ['R5', 'R6']}
]
with open('configs/environment.yaml', 'w') as f:
    yaml.dump(env_cfg, f)
