import os
import json
import csv
import yaml
import numpy as np
import copy
import pandas as pd

from marl.environment.marl_env import MARLResourceAllocationEnv
from marl.algorithms.mappo import MAPPO
from marl.evaluation.evaluator import Evaluator
from marl.utils.seeding import set_all_seeds

from marl.baselines.fcfs import FCFSScheduler
from marl.baselines.round_robin import RoundRobinScheduler
from marl.baselines.met import METScheduler
from marl.baselines.random_feasible import RandomFeasibleScheduler
from marl.baselines.greedy_energy import GreedyEnergyScheduler
from marl.baselines.greedy_sla import GreedySLAScheduler

test_seeds = [1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888, 9999, 10000]

with open('results/best_hyperparameters.yaml', 'r') as f:
    config = yaml.safe_load(f)

per_seed_results = []

methods = {
    'FCFS': FCFSScheduler,
    'RR': RoundRobinScheduler,
    'MET': METScheduler,
    'Random': RandomFeasibleScheduler,
    'GreedyEnergy': GreedyEnergyScheduler,
    'GreedySLA': GreedySLAScheduler
}

# 1. EVALUATE MAPPO
env = MARLResourceAllocationEnv(config)
config['num_agents'] = env.num_agents
config['obs_dim'] = env.get_obs_dim()
config['action_dim'] = env.get_action_dim(env.agent_ids[0])
config['global_state_dim'] = env.get_global_state_dim()

# We test the top seeds, let's just pick the first trained one or average them. 
# Prompt: "evaluate on the held-out test seeds... every result row must contain training_seed".
# I'll evaluate the 3 models (trained on 42, 101, 202) over the 10 test seeds.
for t_seed in [42, 101, 202]:
    model_path = f'results/final_run_seed{t_seed}/best_model.pt'
    if not os.path.exists(model_path):
        continue
        
    algo = MAPPO(config)
    algo.setup(config, config)
    algo.load(model_path)
    
    evaluator = Evaluator(env, None)
    
    for v_seed in test_seeds:
        set_all_seeds(v_seed)
        env.reset(seed=v_seed)
        m = evaluator.evaluate_agents(algo.get_agents())
        
        per_seed_results.append({
            'method': 'MAPPO',
            'config': 'best_hparam',
            'training_seed': t_seed,
            'evaluation_seed': v_seed,
            'checkpoint': 'best_model.pt',
            'rejection_cap': config['rejection']['max_rejection_rate'],
            'completion_rate': m['task_completion_rate_mean'],
            'rejection_rate': m.get('task_rejection_rate_mean', 0.0),
            'throughput': m['throughput_mean'],
            'avg_latency': m.get('avg_task_latency_mean', 0.0),
            'p95_latency': m.get('p95_latency_mean', 0.0),
            'sla_violation_rate': m.get('sla_violation_rate_mean', 0.0),
            'energy_per_completed_task': m.get('energy_per_completed_task_mean', 0.0)
        })

# 2. EVALUATE BASELINES
for name, SchedulerClass in methods.items():
    for v_seed in test_seeds:
        set_all_seeds(v_seed)
        env.reset(seed=v_seed)
        sched = SchedulerClass()
        evaluator = Evaluator(env, None)
        m = evaluator.evaluate_scheduler(sched)
        
        per_seed_results.append({
            'method': name,
            'config': 'baseline',
            'training_seed': 'N/A',
            'evaluation_seed': v_seed,
            'checkpoint': 'N/A',
            'rejection_cap': config['rejection']['max_rejection_rate'],
            'completion_rate': m['task_completion_rate_mean'],
            'rejection_rate': m.get('task_rejection_rate_mean', 0.0),
            'throughput': m['throughput_mean'],
            'avg_latency': m.get('avg_task_latency_mean', 0.0),
            'p95_latency': m.get('p95_latency_mean', 0.0),
            'sla_violation_rate': m.get('sla_violation_rate_mean', 0.0),
            'energy_per_completed_task': m.get('energy_per_completed_task_mean', 0.0)
        })

# Save detailed results
df = pd.DataFrame(per_seed_results)
df.to_csv('results/per_seed_results.csv', index=False)
df.to_json('results/per_seed_results.json', orient='records')

# 3. STATISTICAL SUMMARY -> final_comparison.csv
# Average out MAPPO across training seeds first, then test seeds? 
# Usually, average over all test instances. For MAPPO, we can group by (evaluation_seed) taking the mean across the 3 training seeds, then mean across 10 evaluation seeds. Or just average all 30 evaluations directly.
mappo_df = df[df['method'] == 'MAPPO']
if not mappo_df.empty:
    mappo_agg = mappo_df.groupby('evaluation_seed').mean(numeric_only=True).reset_index()
    mappo_agg['method'] = 'MAPPO'
else:
    mappo_agg = pd.DataFrame()
    
base_df = df[df['method'] != 'MAPPO']
combined_for_stats = pd.concat([mappo_agg, base_df])

summary = combined_for_stats.groupby('method').agg(['mean', 'std'])[
    ['completion_rate', 'rejection_rate', 'throughput', 'avg_latency', 'p95_latency', 'sla_violation_rate', 'energy_per_completed_task']
]

# Format to string "mean ± std"
final_comp = []
for method in summary.index:
    row = {'Method': method}
    for col in summary.columns.levels[0]:
        m_val = summary.loc[method, (col, 'mean')]
        s_val = summary.loc[method, (col, 'std')]
        row[col] = f"{m_val:.3f} ± {s_val:.3f}"
    final_comp.append(row)

final_df = pd.DataFrame(final_comp)
final_df.to_csv('results/final_comparison.csv', index=False)
final_df.to_json('results/final_comparison.json', orient='records')

# Plot comparisons
import matplotlib.pyplot as plt
fig, axes = plt.subplots(3, 2, figsize=(15, 18))
metrics = ['completion_rate', 'throughput', 'avg_latency', 'p95_latency', 'sla_violation_rate', 'energy_per_completed_task']
titles = ['Completion Rate', 'Throughput', 'Average Latency', 'P95 Latency', 'SLA Violation Rate', 'Energy per Completed Task']

for i, (metric, title) in enumerate(zip(metrics, titles)):
    ax = axes[i // 2, i % 2]
    means = [summary.loc[m, (metric, 'mean')] for m in summary.index]
    stds = [summary.loc[m, (metric, 'std')] for m in summary.index]
    ax.bar(summary.index, means, yerr=stds, capsize=5)
    ax.set_title(title)
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('plots/final_comparison.png')
print("Phase 5 evaluation and plotting completed.")
