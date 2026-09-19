import pandas as pd
import json
import yaml
import os
import subprocess

if os.path.exists('results/rejection_ablation.csv'):
    df = pd.read_csv('results/rejection_ablation.csv')
    
    # Filter numeric columns for grouping
    numeric_cols = ['rejection_cap', 'completion_rate', 'rejection_rate', 'sla_violation_rate', 'throughput', 'avg_latency']
    mappo_df = df[df['method'] == 'MAPPO'][numeric_cols].groupby('rejection_cap').mean().reset_index()
    
    best_cap = 0.2
    best_score = -float('inf')
    
    for _, row in mappo_df.iterrows():
        cap = row['rejection_cap']
        comp = row['completion_rate']
        rej = row['rejection_rate']
        sla = row['sla_violation_rate']
        
        score = comp - (0.5 * sla) - (0.1 * rej)
        if score > best_score:
            best_score = score
            best_cap = cap
            
    print(f"Automatically selected best max_rejection_rate: {best_cap}")
    
    with open('configs/environment.yaml', 'r') as f:
        env_cfg = yaml.safe_load(f)
        
    env_cfg['rejection']['max_rejection_rate'] = float(best_cap)
    
    with open('configs/environment.yaml', 'w') as f:
        yaml.dump(env_cfg, f)
        
    with open('configs/reward.yaml', 'r') as f:
        rew_cfg = yaml.safe_load(f)
        
    rew_cfg['admission']['target_rejection_rate'] = max(0.0, float(best_cap) - 0.1)
    
    with open('configs/reward.yaml', 'w') as f:
        yaml.dump(rew_cfg, f)
        
    print("Configs updated. Now executing sweep...")
    subprocess.run(["/Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11", "-u", "scripts/tuning/run_hparam_sweep.py"])
else:
    print("Ablation not finished yet.")
