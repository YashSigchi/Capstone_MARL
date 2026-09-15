import subprocess
import json
import yaml
import numpy as np

print("Running baseline FCFC/MET...")
cmd = ["/Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11", "scripts/run_baselines.py", "--config", "configs/experiment.yaml"]
subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
import pandas as pd
df = pd.read_csv('baseline_comparison.csv', index_col=0)
fcfs_comp = df.loc['fcfs', 'task_completion_rate_mean']
met_comp = df.loc['met', 'task_completion_rate_mean']
print(f"FCFS: {fcfs_comp:.3f}, MET: {met_comp:.3f}")

print("\nRunning Best MAPPO...")
with open('configs/experiment.yaml', 'r') as f:
    cfg = yaml.safe_load(f)
cfg['checkpoint_dir'] = 'checkpoints'
with open('configs/experiment.yaml', 'w') as f:
    yaml.dump(cfg, f)

cmd = ["/Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11", "scripts/train.py", "--config", "configs/experiment.yaml", "--num-episodes", "200"]
subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)

cmd = ["/Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11", "scripts/evaluate.py", "--config", "configs/experiment.yaml", "--checkpoint", "checkpoints/best_model.pt", "--output", "temp.json"]
subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
with open('temp.json', 'r') as f:
    res = json.load(f)
print(f"MAPPO: {res['task_completion_rate_mean']:.3f}")
