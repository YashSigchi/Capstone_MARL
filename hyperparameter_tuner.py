import os
import yaml
import subprocess
import pandas as pd
import json
from copy import deepcopy

BASE_MARL_CONFIG_PATH = 'configs/marl.yaml'
EXP_DIR = 'experiments'
RESULTS_FILE = os.path.join(EXP_DIR, 'tuning_results.csv')
LOGS_DIR = os.path.join(EXP_DIR, 'logs')

def load_base_config():
    with open(BASE_MARL_CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)

def save_config(config, filepath):
    with open(filepath, 'w') as f:
        yaml.dump(config, f)

def run_experiment(run_id, overrides, num_episodes, seeds):
    results = []
    
    for seed in seeds:
        print(f"\n--- Running {run_id} | Seed {seed} ---")
        
        cfg = load_base_config()
        for k, v in overrides.items():
            if k == 'actor_architecture': cfg['actor_hidden_dims'] = v
            elif k == 'critic_architecture': cfg['critic_hidden_dims'] = v
            else: cfg[k] = v
                
        cfg_path = os.path.join(EXP_DIR, 'configs', f"{run_id}_seed{seed}.yaml")
        save_config(cfg, cfg_path)
        
        base_orig = load_base_config()
        save_config(cfg, BASE_MARL_CONFIG_PATH)
        
        with open('configs/experiment.yaml', 'r') as f:
            exp_cfg = yaml.safe_load(f)
        exp_orig = deepcopy(exp_cfg)
        exp_cfg['seed'] = seed
        with open('configs/experiment.yaml', 'w') as f:
            yaml.dump(exp_cfg, f)
            
        cmd = [
            "/Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11", 
            "scripts/train.py", 
            "--config", "configs/experiment.yaml", 
            "--num-episodes", str(num_episodes)
        ]
        
        log_path = os.path.join(LOGS_DIR, f"{run_id}_seed{seed}.log")
        
        try:
            with open(log_path, 'w') as log_f:
                subprocess.run(cmd, check=True, stdout=log_f, stderr=subprocess.STDOUT)
            
            with open('checkpoints/best_model_metadata.json', 'r') as f:
                meta = json.load(f)
                
            metrics = meta['metrics']
            rej_rate = metrics.get('task_rejection_rate', 0.0)
            if 'task_rejection_rate' not in metrics:
                total = metrics.get('completed_tasks', 0) + metrics.get('rejected_tasks', 0) + metrics.get('unfinished_tasks', 0)
                rej_rate = metrics.get('rejected_tasks', 0) / max(1, total)

            results.append({
                'run_id': run_id,
                'training_seed': seed,
                'learning_rate': cfg.get('lr_actor', 3e-4),
                'entropy_coef': cfg.get('entropy_coef', 0.01),
                'gamma': cfg.get('gamma', 0.99),
                'gae_lambda': cfg.get('gae_lambda', 0.95),
                'clip_range': cfg.get('clip_range', 0.2),
                'num_epochs': cfg.get('ppo_epochs', 5),
                'actor_architecture': str(cfg.get('actor_hidden_dims', [64, 64])),
                'critic_architecture': str(cfg.get('critic_hidden_dims', [64, 64])),
                'best_episode': meta.get('training_episode', 0),
                'completion_rate': metrics.get('task_completion_rate', 0.0),
                'rejection_rate': rej_rate,
                'SLA_rate': metrics.get('sla_violation_rate', 0.0),
                'latency': metrics.get('latency', 0.0),
                'throughput': metrics.get('throughput', 0.0),
                'energy_per_task': metrics.get('energy_per_completed_task', 0.0),
                'selection_score': meta.get('selection_score', 0.0)
            })
            
            chk_name = f"best_model_{run_id}_seed{seed}.pt"
            os.rename('checkpoints/best_model.pt', os.path.join(EXP_DIR, 'checkpoints', chk_name))
            
        finally:
            save_config(base_orig, BASE_MARL_CONFIG_PATH)
            with open('configs/experiment.yaml', 'w') as f:
                yaml.dump(exp_orig, f)
                
    return results

def main():
    if os.path.exists(RESULTS_FILE):
        df = pd.read_csv(RESULTS_FILE)
    else:
        df = pd.DataFrame()

    SCREENING_EPISODES = 150
    
    stages = {
        'Baseline': [({'lr_actor': 3e-4, 'lr_critic': 3e-4}, [42, 101, 202])],
        'StageA_LR': [
            ({'lr_actor': 1e-4, 'lr_critic': 1e-4}, [42]),
            ({'lr_actor': 5e-4, 'lr_critic': 5e-4}, [42]),
            ({'lr_actor': 1e-3, 'lr_critic': 1e-3}, [42])
        ],
        'StageB_Entropy': [
            ({'entropy_coef': 0.0}, [42]),
            ({'entropy_coef': 0.001}, [42]),
            ({'entropy_coef': 0.005}, [42]),
            ({'entropy_coef': 0.02}, [42])
        ],
        'StageC_GAE': [
            ({'gamma': 0.95, 'gae_lambda': 0.90}, [42]),
            ({'gamma': 0.995, 'gae_lambda': 0.97}, [42])
        ],
        'StageD_PPO': [
            ({'clip_range': 0.1, 'ppo_epochs': 3}, [42]),
            ({'clip_range': 0.3, 'ppo_epochs': 10}, [42])
        ],
        'StageE_Arch': [
            ({'actor_architecture': [128, 128], 'critic_architecture': [128, 128]}, [42]),
            ({'actor_architecture': [256, 256], 'critic_architecture': [256, 256]}, [42]),
            ({'actor_architecture': [128, 128, 64], 'critic_architecture': [128, 128, 64]}, [42])
        ]
    }
    
    all_results = []
    
    for stage_name, configs in stages.items():
        print(f"\n================ {stage_name} ================")
        for i, (cfg, seeds) in enumerate(configs):
            run_id = f"{stage_name}_cfg{i}"
            
            if not df.empty and run_id in df['run_id'].values and len(df[df['run_id'] == run_id]) >= len(seeds):
                print(f"Skipping {run_id}, already in results.")
                continue
                
            results = run_experiment(run_id, cfg, SCREENING_EPISODES, seeds=seeds)
            all_results.extend(results)
            
            new_df = pd.DataFrame(all_results)
            if df.empty:
                df = new_df
            else:
                df = pd.concat([df, new_df], ignore_index=True)
            df.to_csv(RESULTS_FILE, index=False)
            all_results = []

if __name__ == '__main__':
    main()
