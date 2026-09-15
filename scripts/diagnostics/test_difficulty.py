import subprocess
import yaml
import copy

with open('configs/environment.yaml', 'r') as f:
    base_env = yaml.safe_load(f)
with open('configs/marl.yaml', 'r') as f:
    marl_cfg = yaml.safe_load(f)
with open('configs/experiment.yaml', 'r') as f:
    base_exp = yaml.safe_load(f)

loads = {
    'LOW': {'rate': 0.2, 'cpu_max': 4.0},
    'MEDIUM': {'rate': 1.0, 'cpu_max': 16.0},
    'HIGH': {'rate': 3.0, 'cpu_max': 32.0}
}

for load_name, params in loads.items():
    env_cfg = copy.deepcopy(base_env)
    env_cfg['task_arrival']['rate'] = params['rate']
    env_cfg['task_distributions']['cpu_req']['max'] = params['cpu_max']
    
    with open('configs/environment.yaml', 'w') as f:
        yaml.dump(env_cfg, f)
        
    print(f"\n=== {load_name} LOAD ===")
    subprocess.run(["/Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11", "scripts/evaluation/run_baselines.py", "--config", "configs/experiment.yaml", "--episodes", "5"], check=True)

# Restore
with open('configs/environment.yaml', 'w') as f:
    yaml.dump(base_env, f)
