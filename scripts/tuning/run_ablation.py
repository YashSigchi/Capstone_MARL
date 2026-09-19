import os
import json
import csv
import yaml
import numpy as np
import copy
import sys
import pandas as pd

from marl.environment.marl_env import MARLResourceAllocationEnv
from marl.reward.reward_engine import RewardEngine
from marl.algorithms.mappo import MAPPO
from marl.training.trainer import Trainer
from marl.evaluation.evaluator import Evaluator
from marl.utils.seeding import set_all_seeds
from marl.baselines.met import METScheduler

os.makedirs('results', exist_ok=True)
os.makedirs('plots', exist_ok=True)

def load_config():
    with open('configs/environment.yaml', 'r') as f:
        c = yaml.safe_load(f)
    with open('configs/marl.yaml', 'r') as f:
        c.update(yaml.safe_load(f))
    with open('configs/reward.yaml', 'r') as f:
        r = yaml.safe_load(f)
        c.update({'reward': r}) # merge fully
    return c

config = load_config()

caps = [0.0, 0.1, 0.2, 0.3, 0.4, 1.0]
valid_seeds = [1001, 1002, 1003, 1004, 1005]
training_seed = 42
num_train_eps = 300

results_data = []
rejection_quality_data = []

for cap in caps:
    print(f"\n======================================")
    print(f" ABLATION: max_rejection_rate = {cap}")
    print(f"======================================")
    c = copy.deepcopy(config)
    c['rejection']['max_rejection_rate'] = cap
    c['rejection']['enabled'] = True
    c['reward']['admission']['target_rejection_rate'] = max(0.0, cap - 0.10) # 10% below cap
    
    # 1. TRAIN MAPPO
    set_all_seeds(training_seed)
    env = MARLResourceAllocationEnv(c)
    c['num_agents'] = env.num_agents
    c['obs_dim'] = env.get_obs_dim()
    c['action_dim'] = env.get_action_dim(env.agent_ids[0])
    c['global_state_dim'] = env.get_global_state_dim()
    
    algo = MAPPO(c)
    algo.setup(c, c)
    
    c['num_episodes'] = num_train_eps
    c['eval_interval'] = 100
    c['save_dir'] = f'results/ablation_{cap}'
    
    reward_engine = RewardEngine(c)
    trainer = Trainer(env, algo, reward_engine=reward_engine, config=c)
    trainer.train()
    
    # Load best model
    try:
        algo.load(f'results/ablation_{cap}/best_model.pt')
    except Exception as e:
        print(f"Could not load best model for cap {cap}: {e}")
        
    # 2. EVALUATE MAPPO vs MET
    mappo_evaluator = Evaluator(env, reward_engine, seed=training_seed)
    
    for v_seed in valid_seeds:
        # MAPPO
        set_all_seeds(v_seed)
        env.reset(seed=v_seed)
        m_metric = mappo_evaluator.evaluate_agents(algo.get_agents(), num_episodes=1, deterministic=True)
        
        # Rejection Quality (just grab from last eval ep)
        if v_seed == valid_seeds[-1]:
            for t in env._rejected_task_records:
                rejection_quality_data.append({
                    'cap': cap,
                    'status': 'rejected',
                    'work_units': t.get('work_units', 0),
                    'cpu_req': t.get('cpu_req', 0),
                    'memory_req': t.get('memory_req', 0),
                    'deadline': t.get('deadline', 0),
                    'feasible_resources': t.get('feasible_resource_count', 0),
                    'est_exec_time': t.get('min_exec_time', 0)
                })
            for t in env._completed_tasks:
                rejection_quality_data.append({
                    'cap': cap,
                    'status': 'completed',
                    'work_units': getattr(t, 'work_units', 0),
                    'cpu_req': getattr(t, 'cpu_req', 0),
                    'memory_req': getattr(t, 'memory_req', 0),
                    'deadline': getattr(t, 'deadline', 0),
                    'feasible_resources': 0,
                    'est_exec_time': 0
                })

        # MET
        met_agent = METScheduler()
        met_eval = Evaluator(env, reward_engine, seed=training_seed)
        met_metric = met_eval.evaluate_scheduler(met_agent, num_episodes=1)
        
        results_data.append({
            'run_id': f"mappo_{cap}_{v_seed}",
            'method': 'MAPPO',
            'rejection_cap': cap,
            'training_seed': training_seed,
            'validation_seed': v_seed,
            'completion_rate': m_metric.get('task_completion_rate_mean', 0.0),
            'rejection_rate': m_metric.get('task_rejection_rate_mean', 0.0),
            'throughput': m_metric.get('throughput_mean', 0.0),
            'avg_latency': m_metric.get('avg_task_latency_mean', 0.0),
            'sla_violation_rate': m_metric.get('sla_violation_rate_mean', 0.0),
            'energy_per_completed_task': m_metric.get('energy_per_completed_task_mean', 0.0)
        })
        
        results_data.append({
            'run_id': f"met_{cap}_{v_seed}",
            'method': 'MET',
            'rejection_cap': cap,
            'training_seed': training_seed,
            'validation_seed': v_seed,
            'completion_rate': met_metric.get('task_completion_rate_mean', 0.0),
            'rejection_rate': met_metric.get('task_rejection_rate_mean', 0.0),
            'throughput': met_metric.get('throughput_mean', 0.0),
            'avg_latency': met_metric.get('avg_task_latency_mean', 0.0),
            'sla_violation_rate': met_metric.get('sla_violation_rate_mean', 0.0),
            'energy_per_completed_task': met_metric.get('energy_per_completed_task_mean', 0.0)
        })

df = pd.DataFrame(results_data)
# Convert single element lists to float if any slipped through
for col in ['completion_rate', 'rejection_rate', 'throughput', 'avg_latency', 'sla_violation_rate', 'energy_per_completed_task']:
    df[col] = df[col].apply(lambda x: x[0] if isinstance(x, list) else x)

df.to_csv('results/rejection_ablation.csv', index=False)
rq_df = pd.DataFrame(rejection_quality_data)
rq_df.to_csv('results/rejection_quality.csv', index=False)

# Analyze and print summary
print("\n=== SUMMARY MAPPO vs MET ===")
numeric_cols = ['completion_rate', 'rejection_rate', 'sla_violation_rate', 'throughput', 'avg_latency']
summary = df.groupby(['rejection_cap', 'method'])[numeric_cols].mean().reset_index()
print(summary)

# Plotting using matplotlib
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

mappo_df = df[df['method'] == 'MAPPO'].groupby('rejection_cap').mean().reset_index()
met_df = df[df['method'] == 'MET'].groupby('rejection_cap').mean().reset_index()

axes[0,0].plot(mappo_df['rejection_cap'], mappo_df['completion_rate'], label='MAPPO', marker='o')
axes[0,0].plot(met_df['rejection_cap'], met_df['completion_rate'], label='MET', marker='s', linestyle='--')
axes[0,0].set_title('Completion Rate vs Rejection Cap')
axes[0,0].legend()

axes[0,1].plot(mappo_df['rejection_cap'], mappo_df['sla_violation_rate'], label='MAPPO', marker='o')
axes[0,1].plot(met_df['rejection_cap'], met_df['sla_violation_rate'], label='MET', marker='s', linestyle='--')
axes[0,1].set_title('SLA Violation Rate vs Rejection Cap')

axes[1,0].plot(mappo_df['rejection_cap'], mappo_df['throughput'], label='MAPPO', marker='o')
axes[1,0].plot(met_df['rejection_cap'], met_df['throughput'], label='MET', marker='s', linestyle='--')
axes[1,0].set_title('Throughput vs Rejection Cap')

axes[1,1].plot(mappo_df['rejection_cap'], mappo_df['avg_latency'], label='MAPPO', marker='o')
axes[1,1].plot(met_df['rejection_cap'], met_df['avg_latency'], label='MET', marker='s', linestyle='--')
axes[1,1].set_title('Average Latency vs Rejection Cap')

plt.tight_layout()
plt.savefig('plots/rejection_tradeoff.png')
print("Saved plots/rejection_tradeoff.png")

