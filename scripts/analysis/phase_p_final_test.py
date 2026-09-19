import os
import pandas as pd
import numpy as np
from marl.environment.marl_env import MARLResourceAllocationEnv
from marl.reward.reward_engine import RewardEngine
from marl.algorithms.mappo import MAPPO
from marl.evaluation.evaluator import Evaluator

from marl.baselines.fcfs import FCFSScheduler
from marl.baselines.round_robin import RoundRobinScheduler
from marl.baselines.met import METScheduler
from marl.baselines.random_feasible import RandomFeasibleScheduler
from marl.baselines.greedy_energy import GreedyEnergyScheduler
from marl.baselines.greedy_sla import GreedySLAScheduler
from marl.utils.seeding import set_all_seeds
import yaml

with open('results/best_hyperparameters.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Use the best seed
best_train_seed = 42

env = MARLResourceAllocationEnv(config)
config['num_agents'] = env.num_agents
config['obs_dim'] = env.get_obs_dim()
config['action_dim'] = env.get_action_dim(env.agent_ids[0])
config['global_state_dim'] = env.get_global_state_dim()

algo = MAPPO(config)
algo.setup(config, config)
algo.load(f'results/final_run_seed{best_train_seed}/best_model.pt')

reward_engine = RewardEngine(config)


test_seeds = [1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888, 9999, 10000]

schedulers = {
    'MAPPO': algo.get_agents(),
    'FCFS': FCFSScheduler(),
    'RR': RoundRobinScheduler(),
    'MET': METScheduler(),
    'Random': RandomFeasibleScheduler(),
    'GreedyEnergy': GreedyEnergyScheduler(),
    'GreedySLA': GreedySLAScheduler()
}

results = []
for name, policy in schedulers.items():
    print(f"Evaluating {name}...")
    for ts in test_seeds:
        set_all_seeds(ts)
        env.reset(seed=ts)
        
        
        evaluator = Evaluator(env, reward_engine, seed=ts)
        if name == 'MAPPO':

            m = evaluator.evaluate_agents(policy, num_episodes=1, deterministic=True)
        else:
            m = evaluator.evaluate_scheduler(policy, num_episodes=1)
            
        results.append({
            'method': name,
            'test_seed': ts,
            'completion_rate': m.get('task_completion_rate_mean', 0.0),
            'rejection_rate': m.get('task_rejection_rate_mean', 0.0),
            'throughput': m.get('throughput_mean', 0.0),
            'avg_latency': m.get('avg_task_latency_mean', 0.0),
            'sla_violation_rate': m.get('sla_violation_rate_mean', 0.0),
            'energy_per_task': m.get('energy_per_completed_task_mean', 0.0)
        })

df = pd.DataFrame(results)
df.to_csv('results/phase_p_results.csv', index=False)

summary = df.groupby('method').agg(['mean', 'std']).reset_index()
print(summary)
