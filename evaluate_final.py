import subprocess
import yaml
import os
import json
import numpy as np

EXP_DIR = 'experiments'
seeds = [42, 101, 202]
test_seeds = [301, 302, 303]

mappo_results = []
baseline_results = {
    'fcfs': [], 'met': [], 'round_robin': []
}

for seed in seeds:
    chk_dir = os.path.join(EXP_DIR, 'checkpoints', f'run_seed{seed}')
    chk_path = os.path.join(chk_dir, 'best_model.pt')
    
    for t_seed in test_seeds:
        # Run MAPPO eval
        cmd = [
            "/Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11", 
            "scripts/evaluate.py", 
            "--config", "configs/experiment.yaml", 
            "--checkpoint", chk_path,
            "--episodes", "10",
            "--seed", str(t_seed),
            "--output", "temp_eval.json"
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        
        with open("temp_eval.json", 'r') as f:
            metrics = json.load(f)
            mappo_results.append(metrics)

# Run Baselines on same test seeds
for t_seed in test_seeds:
    cmd = [
        "/Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11", 
        "scripts/run_baselines.py", 
        "--config", "configs/experiment.yaml", 
        "--episodes", "10",
        "--seed", str(t_seed)
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    
    import pandas as pd
    df = pd.read_csv('baseline_comparison.csv', index_col=0)
    for b in ['fcfs', 'met', 'round_robin']:
        if b in df.index:
            baseline_results[b].append({
                'task_completion_rate_mean': df.loc[b, 'task_completion_rate_mean'],
                'sla_violation_rate_mean': df.loc[b, 'sla_violation_rate_mean']
            })

print("\n=== FINAL UNSEEN TEST PERFORMANCE (3 training seeds x 3 test seeds = 9 evals) ===")
mappo_comp = [m['task_completion_rate_mean'] for m in mappo_results]
mappo_sla = [m['sla_violation_rate_mean'] for m in mappo_results]
print(f"MAPPO Completion Rate: {np.mean(mappo_comp):.3f} +/- {np.std(mappo_comp):.3f}")
print(f"MAPPO SLA Violation : {np.mean(mappo_sla):.3f} +/- {np.std(mappo_sla):.3f}")

for b in ['fcfs', 'met', 'round_robin']:
    comp = [m['task_completion_rate_mean'] for m in baseline_results[b]]
    sla = [m['sla_violation_rate_mean'] for m in baseline_results[b]]
    print(f"{b.upper()} Completion Rate: {np.mean(comp):.3f} +/- {np.std(comp):.3f}")
    print(f"{b.upper()} SLA Violation : {np.mean(sla):.3f} +/- {np.std(sla):.3f}")
