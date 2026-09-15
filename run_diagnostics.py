import subprocess
import yaml

with open('configs/environment.yaml', 'r') as f:
    env_cfg = yaml.safe_load(f)

env_cfg['num_resources'] = 6
env_cfg['resources'] = [
    {'id': f'R{i}', 'cpu_capacity': 16, 'memory_capacity': 32, 'gpu_capacity': 1 if i%2==0 else 0,
     'power_rate': 250, 'processing_speed': 1.0 + (i*0.2), 'network_bandwidth': 10.0}
    for i in range(1, 7)
]
env_cfg['clusters'] = [
    {'id': 'C1', 'resource_ids': ['R1', 'R2']},
    {'id': 'C2', 'resource_ids': ['R3', 'R4']},
    {'id': 'C3', 'resource_ids': ['R5', 'R6']}
]
env_cfg['max_steps'] = 100
env_cfg['task_arrival']['rate'] = 1.0

with open('configs/environment.yaml', 'w') as f:
    yaml.dump(env_cfg, f)

with open('configs/marl.yaml', 'r') as f:
    marl_cfg = yaml.safe_load(f)

marl_cfg['num_agents'] = 3
marl_cfg['agent_mapping'] = {0: 'C1', 1: 'C2', 2: 'C3'}

with open('configs/marl.yaml', 'w') as f:
    yaml.dump(marl_cfg, f)

with open('configs/experiment.yaml', 'r') as f:
    exp_cfg = yaml.safe_load(f)

exp_cfg['num_episodes'] = 50
exp_cfg['steps_per_episode'] = 100
exp_cfg['eval_interval'] = 10
exp_cfg['save_interval'] = 20
exp_cfg['checkpoint_dir'] = 'checkpoints'

with open('configs/experiment.yaml', 'w') as f:
    yaml.dump(exp_cfg, f)

print("--- Running Training ---")
subprocess.run(["/Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11", "scripts/train.py", "--config", "configs/experiment.yaml", "--num-episodes", "50"], check=True)

print("\n--- Running Evaluation ---")
subprocess.run(["/Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11", "scripts/evaluate.py", "--config", "configs/experiment.yaml", "--checkpoint", "checkpoints/best_model.pt", "--episodes", "10"], check=True)

print("\n--- Running Baselines ---")
subprocess.run(["/Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11", "scripts/run_baselines.py", "--config", "configs/experiment.yaml", "--episodes", "10"], check=True)
