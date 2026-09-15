import yaml

with open('configs/environment.yaml', 'r') as f:
    env_cfg = yaml.safe_load(f)

env_cfg['topology']['type'] = 'fully_connected'

with open('configs/environment.yaml', 'w') as f:
    yaml.dump(env_cfg, f)
