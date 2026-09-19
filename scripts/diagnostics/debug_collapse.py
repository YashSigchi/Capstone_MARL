import os
import json
import yaml
import numpy as np
import pandas as pd
from pprint import pprint

from marl.environment.marl_env import MARLResourceAllocationEnv
from marl.baselines.met import METScheduler
from marl.baselines.fcfs import FCFSScheduler

os.makedirs('results', exist_ok=True)

with open('configs/environment.yaml', 'r') as f:
    env_cfg = yaml.safe_load(f)
with open('configs/reward.yaml', 'r') as f:
    env_cfg.update(yaml.safe_load(f))
with open('configs/marl.yaml', 'r') as f:
    env_cfg.update(yaml.safe_load(f))

# Ensure rejection is fully enabled
env_cfg['rejection']['enabled'] = True
env_cfg['rejection']['max_rejection_rate'] = 1.0

env = MARLResourceAllocationEnv(env_cfg)
from marl.reward.reward_engine import RewardEngine
reward_engine = RewardEngine(env_cfg)

obs_dict, _ = env.reset(seed=42)

aid = env.agent_ids[0]
action_dim = env.get_action_dim(aid)

print("=== TASK 4: ACTIVE CONFIGURATION ===")
print(f"rejection.enabled: {env_cfg['rejection']['enabled']}")
print(f"rejection.max_rejection_rate: {env_cfg['rejection']['max_rejection_rate']}")
print(f"reward.admission (target_rejection_rate expected, check if exists): {env_cfg.get('admission', {})}")
print(f"action_penalty: {env_cfg.get('action_penalty', -0.1)}")
print(f"Actor Output Dimension vs Env action_dim: {action_dim} == resources + 2")
print("="*50)

# TASK 9 & 8: ACTION MASKING AND FEASIBILITY
print("\n=== TASK 8 & 9: FEASIBILITY & MASKING ===")
feasibility_counts = []
mask_records = []
obs_records = []

obs, _ = env.reset(seed=42)
for _ in range(100):
    aid = list(obs.keys())[0]
    mask = env.get_action_mask(aid)
    active_task = env.get_current_task(aid)
    
    cluster_id = env.agent_cluster_map[aid]
    cluster = env.clusters[cluster_id]
    res_count = cluster.num_resources
    
    valid_res = int(np.sum(mask[:res_count]))
    valid_defer = int(mask[-2])
    valid_reject = int(mask[-1])
    feasibility_counts.append(valid_res)
    
    if len(mask_records) < 5:
        mask_records.append({
            'res_mask': mask[:res_count].tolist(),
            'defer': valid_defer,
            'reject': valid_reject
        })
        obs_records.append(obs[aid])
        
    # Just take random valid action
    valid_acts = np.where(mask == 1)[0]
    if len(valid_acts) > 0:
        action = np.random.choice(valid_acts)
    else:
        action = res_count  # defer
    obs, rewards, done, truncated, _ = env.step({aid: action})
    if done[aid]:
        env.reset(seed=42)

feasibility_dist = {k: feasibility_counts.count(k) for k in set(feasibility_counts)}
print(f"Feasibility Dist over 100 states: {feasibility_dist}")
print(f"Sample Mask: {mask_records[0]}")

# TASK 12 & 13: OBSERVATION FEATURES
print("\n=== TASK 12 & 13: OBSERVATION FEATURES ===")
print("Observation shape:", obs_records[0].shape)
# I will print the raw values later if needed. 

# TASK 1 & 2: REWARD RECONSTRUCTION
print("\n=== TASK 1 & 2: REWARD DECOMPOSITION ===")

def run_policy(env_ref, policy_type):
    env_ref.reset(seed=101)
    ep_reward = 0.0
    reward_components_sum = {
        'action_penalty': 0.0,
        'admission': 0.0,
        'sla': 0.0,
        'latency': 0.0,
        'queue': 0.0,
        'energy': 0.0,
        'utilization': 0.0,
        'load_balance': 0.0,
        'throughput': 0.0
    }
    
    num_defers = 0
    num_rejects = 0
    num_allocs = 0
    
    done = False
    step = 0
    while not done and step < 200:
        aid = list(env_ref.agent_ids)[0]
        mask = env_ref.get_action_mask(aid)
        cluster_id = env_ref.agent_cluster_map[aid]
        res_count = env_ref.clusters[cluster_id].num_resources
        
        valid_acts = np.where(mask == 1)[0]
        if len(valid_acts) == 0:
            act = res_count # defer fallback
        elif policy_type == 'REJECT':
            act = res_count + 1 if mask[-1] == 1 else res_count
        elif policy_type == 'DEFER':
            act = res_count if mask[-2] == 1 else valid_acts[0]
        elif policy_type == 'FCFS':
            res_acts = valid_acts[valid_acts < res_count]
            if len(res_acts) > 0: act = res_acts[0]
            else: act = res_count
        
        if act == res_count + 1: num_rejects += 1
        elif act == res_count: num_defers += 1
        else: num_allocs += 1
            
        obs, _, dones, _, infos = env_ref.step({aid: act})
        reward_info = infos.get(aid, {}).get('reward_info', {})
        result = reward_engine.compute(reward_info)
        
        ep_reward += result.total
        info = result.components
        for k in reward_components_sum.keys():
            reward_components_sum[k] += info.get(f'{k}', 0.0)
            
        done = dones[aid]
        step += 1
        
    return {
        'policy': policy_type,
        'episode_reward': ep_reward,
        'completed': len(env_ref._completed_tasks),
        'rejected': len(env_ref._rejected_tasks),
        'unfinished': sum(q.length for q in env_ref.queues.values()),
        'components': reward_components_sum,
        'defers': num_defers,
        'allocs': num_allocs,
        'steps': step
    }

res_reject = run_policy(env, 'REJECT')
res_defer = run_policy(env, 'DEFER')
res_fcfs = run_policy(env, 'FCFS')

print(f"REJECT Policy: Reward: {res_reject['episode_reward']} | Rej: {res_reject['rejected']} | Comp: {res_reject['completed']}")
print(f"DEFER Policy: Reward: {res_defer['episode_reward']} | Rej: {res_defer['rejected']} | Comp: {res_defer['completed']}")
print(f"FCFS Policy: Reward: {res_fcfs['episode_reward']} | Rej: {res_fcfs['rejected']} | Comp: {res_fcfs['completed']}")

with open('results/reward_breakdown_full.json', 'w') as f:
    json.dump({'REJECT': res_reject, 'DEFER': res_defer, 'FCFS': res_fcfs}, f, indent=2)

df = pd.DataFrame([{
    'policy': r['policy'], 'reward': r['episode_reward'], 'completed': r['completed'], 
    'rejected': r['rejected'], 'unfinished': r['unfinished']
} for r in [res_reject, res_defer, res_fcfs]])
df.to_csv('results/action_strategy_comparison.csv', index=False)
print("Saved comparison and breakdown.")
