import os
import json
import csv
import yaml
import numpy as np
import copy
from typing import Dict, List, Any

from marl.environment.marl_env import MARLResourceAllocationEnv
from marl.reward.reward_engine import RewardEngine
from marl.environment.task import TaskRepresentation
from marl.training.trainer import Trainer
from marl.algorithms.mappo import MAPPO
from marl.baselines.met import METScheduler

os.makedirs('results', exist_ok=True)

# ---------------------------------------------------------
# SETUP
# ---------------------------------------------------------
def load_config():
    with open('configs/environment.yaml', 'r') as f:
        env_cfg = yaml.safe_load(f)
    with open('configs/marl.yaml', 'r') as f:
        marl_cfg = yaml.safe_load(f)
    with open('configs/reward.yaml', 'r') as f:
        r_cfg = yaml.safe_load(f)
    config = dict(env_cfg)
    config.update(marl_cfg)
    return config, r_cfg

config, r_cfg = load_config()

# ---------------------------------------------------------
# 4. VERIFY REWARD VALUES FOR IDENTICAL STATES
# ---------------------------------------------------------
print("--- 4. Verify Identical States ---")
engine = RewardEngine(r_cfg)

def test_identical_state(rejection_rate: float, max_rej_rate: float, action_type: str):
    info = {
        'rejection_enabled': True,
        'current_rejection_rate': rejection_rate,
        'max_rejection_rate': max_rej_rate,
        'action_was_reject': action_type == 'reject',
        'action_was_defer': action_type == 'defer',
        'system_avg_utilization': 0.5,
        'all_utilizations': [0.5, 0.5],
        'sla_violated': False,
        'sla_slack': 10.0,
        'task_priority': 3,
        'local_queue_length': 1,
        'total_completed_tasks': 10,
        'total_tasks_this_step': 20,
    }
    # For latency/energy, just put dummies if resource alloc
    if action_type == 'alloc':
        info['total_completion_time'] = 5.0
        info['energy_consumed'] = 50.0
        
    res = engine.compute(info)
    return res

rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]
for r in rates:
    res = test_identical_state(r, 0.2, 'reject')
    print(f"Reject @ {r:.2f} rate (Cap 0.2): Adm={res.components.get('admission',0.0):.3f} | Pen={res.components.get('action_penalty',0.0):.3f} | Total={res.total:.3f}")

alloc_res = test_identical_state(0.0, 0.2, 'alloc')
defer_res = test_identical_state(0.0, 0.2, 'defer')
print(f"Alloc Total: {alloc_res.total:.3f}")
print(f"Defer Total: {defer_res.total:.3f}")

# ---------------------------------------------------------
# 5. TEST THE "DIFFICULT TASK" CASE
# ---------------------------------------------------------
print("\n--- 5. Test Difficult Task vs Easy Task ---")
# Difficult task: high work, tight deadline
hard_info = {
    'action_was_reject': False, 'action_was_defer': False, 'rejection_enabled': True,
    'current_rejection_rate': 0.0, 'max_rejection_rate': 0.2, 'system_avg_utilization': 0.5,
    'all_utilizations': [0.5, 0.5], 'total_completed_tasks': 10, 'total_tasks_this_step': 20,
    'sla_violated': True, 'sla_slack': -5.0, 'task_priority': 5, 'local_queue_length': 5,
    'total_completion_time': 50.0, 'energy_consumed': 200.0, 
    'allocated_task': TaskRepresentation('T1', 8.0, 32.0, 1.0, 100.0, 10.0, 5, 100.0, 1.0, [], 0, 'gold')
}

easy_info = dict(hard_info)
easy_info.update({
    'sla_violated': False, 'sla_slack': 20.0, 'task_priority': 1, 'local_queue_length': 0,
    'total_completion_time': 5.0, 'energy_consumed': 10.0,
    'allocated_task': TaskRepresentation('T2', 1.0, 1.0, 0.0, 5.0, 30.0, 1, 1.0, 0.0, [], 0, 'bronze')
})

hard_alloc = engine.compute(hard_info)
hard_rej_info = dict(hard_info); hard_rej_info.update({'action_was_reject': True, 'sla_violated': False, 'energy_consumed': 0})
hard_rej = engine.compute(hard_rej_info)

easy_alloc = engine.compute(easy_info)
easy_rej_info = dict(easy_info); easy_rej_info.update({'action_was_reject': True, 'sla_violated': False, 'energy_consumed': 0})
easy_rej = engine.compute(easy_rej_info)

print(f"HARD TASK: Alloc={hard_alloc.total:.3f} | Reject={hard_rej.total:.3f}")
print(f"EASY TASK: Alloc={easy_alloc.total:.3f} | Reject={easy_rej.total:.3f}")


# ---------------------------------------------------------
# 6 & 7 & 9. REWARD COMPONENT DIAGNOSTICS & REJECTION QUALITY
# ---------------------------------------------------------
print("\n--- Running Deterministic Policy Diagnostics ---")

def run_policy(env, policy_type, num_episodes=5):
    all_metrics = []
    all_reward_components = []
    
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            actions = {}
            for aid in env.agent_ids:
                mask = env.get_action_mask(aid)
                # mask: [R1, R2, ..., DEFER, REJECT]
                valid_acts = np.where(mask == 1.0)[0]
                if len(valid_acts) == 0:
                    actions[aid] = len(mask) - 2 # defer
                    continue
                
                if policy_type == 'reject' and mask[-1] == 1.0:
                    a = len(mask) - 1
                elif policy_type == 'defer' and mask[-2] == 1.0:
                    a = len(mask) - 2
                elif policy_type == 'first':
                    a = valid_acts[0]
                elif policy_type == 'random':
                    a = np.random.choice(valid_acts)
                else:
                    a = valid_acts[0]
                actions[aid] = int(a)
                
            obs, rewards, term, trunc, infos = env.step(actions)
            done = all(term.values()) or all(trunc.values())
            
            # collect reward components
            if policy_type == 'random': # only do deep component diagnostics on random
                for aid, info in infos.items():
                    if 'reward_info' in info:
                        comp = engine.compute(info['reward_info']).components
                        all_reward_components.append(comp)

        # removed evaluator.
        all_metrics.append({})
        
    return all_metrics, all_reward_components

random_env = MARLResourceAllocationEnv(config)
rand_metrics, rand_comps = run_policy(random_env, 'random', num_episodes=10)

# Save component diagnostics
comp_keys = ['latency', 'energy', 'utilization', 'load_balance', 'sla', 'queue', 'throughput', 'admission', 'action_penalty']
diag_data = []
for k in comp_keys:
    vals = [c.get(k, 0.0) for c in rand_comps if k in c]
    if not vals: continue
    diag_data.append({
        'component': k,
        'mean': np.mean(vals),
        'std': np.std(vals),
        'min': np.min(vals),
        'max': np.max(vals)
    })

with open('results/reward_component_diagnostics.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['component', 'mean', 'std', 'min', 'max'])
    writer.writeheader()
    writer.writerows(diag_data)

with open('results/reward_component_diagnostics.json', 'w') as f:
    json.dump(diag_data, f, indent=2)

# Save Rejection Quality
rej_records = random_env._rejected_task_records
comp_tasks = random_env._completed_tasks

print(f"Collected {len(rej_records)} rejected tasks and {len(comp_tasks)} completed tasks.")

def mean_attr(tasks, attr, is_dict=False):
    if len(tasks) == 0: return 0.0
    if is_dict:
        return np.mean([t.get(attr, 0.0) for t in tasks])
    else:
        return np.mean([getattr(t, attr, 0.0) for t in tasks])

qual_data = []
attrs = ['work_units', 'cpu_req', 'memory_req', 'gpu_req', 'deadline', 'priority']
for attr in attrs:
    qual_data.append({
        'feature': attr,
        'rejected_mean': mean_attr(rej_records, attr, True),
        'completed_mean': mean_attr(comp_tasks, attr, False)
    })
# Specific ones
qual_data.append({
    'feature': 'feasible_resources',
    'rejected_mean': mean_attr(rej_records, 'feasible_resource_count', True),
    'completed_mean': 0.0 # Not tracked for completed
})
qual_data.append({
    'feature': 'estimated_exec_time',
    'rejected_mean': mean_attr(rej_records, 'min_exec_time', True),
    'completed_mean': 0.0 # Hard to back-calculate here exactly
})

with open('results/rejection_quality.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['feature', 'rejected_mean', 'completed_mean'])
    writer.writeheader()
    writer.writerows(qual_data)

# ---------------------------------------------------------
# 10. ABLATION ON CAPS (100 EPISODES PER CAP)
# ---------------------------------------------------------
print("\n--- 10. Ablation ---")

caps = [0.0, 0.1, 0.2, 0.3, 0.4, 1.0]
tradeoff_results = []

for cap in caps:
    print(f"\nTraining cap={cap} for 100 episodes...")
    c = copy.deepcopy(config)
    c['rejection']['max_rejection_rate'] = cap
    c['rejection']['enabled'] = (cap > 0.0)
    
    env_cap = MARLResourceAllocationEnv(c)
    
    c['num_agents'] = env_cap.num_agents
    c['obs_dim'] = env_cap.get_obs_dim()
    c['action_dim'] = env_cap.get_action_dim(env_cap.agent_ids[0])
    c['global_state_dim'] = env_cap.get_global_state_dim()
    
    algo = MAPPO(c)
    algo.setup(c, c)
    
    c['training'] = c.get('training', {})
    c['training']['num_episodes'] = 100
    c['training']['eval_interval'] = 100
    c['training']['save_dir'] = f'results/ablation_{cap}'
    
    trainer = Trainer(env_cap, algo, config=c)
    trainer.train()
    
    # Eval the best
    from marl.evaluation.evaluator import Evaluator
    from marl.utils.seeding import set_all_seeds
    algo.load(f'results/ablation_{cap}/best_model.pt')
    
    evaluator = Evaluator(env_cap, algo)
    mappo_metrics = []
    met_metrics = []
    
    # 5 Test seeds
    for s in [1111, 2222, 3333, 4444, 5555]:
        set_all_seeds(s)
        env_cap.reset(seed=s)
        met_agent = METScheduler(env_cap)
        met_eval = Evaluator(env_cap, None)
        
        # Eval MAPPO
        env_cap.reset(seed=s)
        mappo_m = evaluator.evaluate_agents(algo.get_agents())
        mappo_metrics.append(mappo_m)
        
        # Eval MET
        env_cap.reset(seed=s)
        met_m = met_eval.evaluate_scheduler(met_agent)
        met_metrics.append(met_m)
        
    m_comp = np.mean([m.task_completion_rate for m in mappo_metrics])
    m_rej = np.mean([m.task_rejection_rate for m in mappo_metrics])
    m_lat = np.mean([m.avg_task_latency for m in mappo_metrics])
    m_thr = np.mean([m.throughput for m in mappo_metrics])
    m_sla = np.mean([m.sla_violation_rate for m in mappo_metrics])
    
    met_comp = np.mean([m.task_completion_rate for m in met_metrics])
    met_thr = np.mean([m.throughput for m in met_metrics])
    met_lat = np.mean([m.avg_task_latency for m in met_metrics])
    met_sla = np.mean([m.sla_violation_rate for m in met_metrics])
    
    tradeoff_results.append({
        'cap': cap,
        'mappo_completion': m_comp,
        'mappo_rejection': m_rej,
        'mappo_latency': m_lat,
        'mappo_throughput': m_thr,
        'mappo_sla': m_sla,
        'met_completion': met_comp,
        'met_throughput': met_thr,
        'met_latency': met_lat,
        'met_sla': met_sla,
        'diff_completion': m_comp - met_comp
    })
    
with open('results/rejection_tradeoff.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=[
        'cap', 'mappo_completion', 'mappo_rejection', 'mappo_latency', 'mappo_throughput', 'mappo_sla',
        'met_completion', 'met_throughput', 'met_latency', 'met_sla', 'diff_completion'
    ])
    writer.writeheader()
    writer.writerows(tradeoff_results)

print("Diagnostic script complete.")
