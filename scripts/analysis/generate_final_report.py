import os
import json
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from marl.environment.marl_env import MARLResourceAllocationEnv
from marl.reward.reward_engine import RewardEngine
from marl.algorithms.mappo import MAPPO
from marl.evaluation.evaluator import Evaluator
from marl.utils.seeding import set_all_seeds
import copy

os.makedirs('results', exist_ok=True)
os.makedirs('plots', exist_ok=True)

with open('results/best_hyperparameters.yaml', 'r') as f:
    config = yaml.safe_load(f)

seeds = [42, 101, 202]
valid_seeds = [1001, 1002, 1003, 1004, 1005]
final_results = []
metrics_cache = {}

for s in seeds:
    c = copy.deepcopy(config)
    env = MARLResourceAllocationEnv(c)
    c['num_agents'] = env.num_agents
    c['obs_dim'] = env.get_obs_dim()
    c['action_dim'] = env.get_action_dim(env.agent_ids[0])
    c['global_state_dim'] = env.get_global_state_dim()
    
    algo = MAPPO(c)
    algo.setup(c, c)
    model_path = f'results/final_run_seed{s}/best_model.pt'
    
    try:
        algo.load(model_path)
    except Exception as e:
        print(f"Skipping seed {s}: {e}")
        continue
        
    reward_engine = RewardEngine(c)
    evaluator = Evaluator(env, reward_engine, seed=s)
    
    seed_metrics = []
    for vs in valid_seeds:
        set_all_seeds(vs)
        env.reset(seed=vs)
        m = evaluator.evaluate_agents(algo.get_agents(), num_episodes=1, deterministic=True)
        seed_metrics.append({
            'train_seed': s,
            'valid_seed': vs,
            'completion_rate': m.get('task_completion_rate_mean', 0.0),
            'rejection_rate': m.get('task_rejection_rate_mean', 0.0),
            'throughput': m.get('throughput_mean', 0.0),
            'avg_latency': m.get('avg_task_latency_mean', 0.0),
            'sla_violation_rate': m.get('sla_violation_rate_mean', 0.0)
        })
    
    final_results.extend(seed_metrics)
    metrics_cache[s] = seed_metrics

if final_results:
    df = pd.DataFrame(final_results)
    df.to_csv('results/per_seed_results.csv', index=False)
    
    with open('results/per_seed_results.json', 'w') as f:
        json.dump(metrics_cache, f, indent=2)
        
    print("Final evaluation written to CSV and JSON.")
else:
    print("No final results to write.")

# Plotting Training Curves
# NOTE: Trainer does not persist history.json. If none is found, skip plotting so we
# never overwrite plots/training_curve.png with an empty chart. To build the curve from
# a training stdout log use: python scripts/analysis/plot_training_curve.py
plt.figure(figsize=(10, 6))
plotted = False
for s in seeds:
    history_file = f'results/final_run_seed{s}/history.json'
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            h = json.load(f)
            if 'episode_reward' in h:
                # Smooth the curve
                rew = pd.Series(h['episode_reward']).rolling(10).mean()
                plt.plot(rew, label=f'Seed {s}')
                plotted = True

if plotted:
    plt.title('MAPPO Training Curve (Top Config)')
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward (Smoothed)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/training_curve.png')
    print("Saved plots/training_curve.png")
else:
    print("No history.json found; skipped training curve "
          "(use scripts/analysis/plot_training_curve.py).")

