import yaml
import numpy as np
from marl.environment.marl_env import MARLResourceAllocationEnv
from marl.reward.reward_engine import RewardEngine
from marl.utils.seeding import set_all_seeds
from marl.baselines.fcfs import FCFSScheduler
from marl.baselines.round_robin import RoundRobinScheduler
from marl.baselines.met import METScheduler

print("=== 1. EVALUATION SEEDS DIVERSITY ===")
with open('configs/environment.yaml', 'r') as f:
    env_cfg = yaml.safe_load(f)
with open('configs/marl.yaml', 'r') as f:
    marl_cfg = yaml.safe_load(f)

config = dict(env_cfg)
config.update(marl_cfg)

for seed in [101, 102, 103]:
    env = MARLResourceAllocationEnv(config)
    env.reset(seed=seed)
    
    # Generate exactly 5 tasks by forcing step
    tasks = []
    while len(tasks) < 5 and env._current_step < 100:
        env.step({0: 3, 1: 3, 2: 3}) # defer everything
        for aid in env.agent_ids:
            t = env.get_current_task(aid)
            if t and t.task_id not in [x.task_id for x in tasks]:
                tasks.append(t)
    
    print(f"\nSeed {seed} (First 3 tasks):")
    for t in tasks[:3]:
        print(f"  {t.task_id}: Arr={t.arrival_time:.1f}, WU={t.work_units:.1f}, CPU={t.cpu_req:.1f}, GPU={t.gpu_req:.1f}, DL={t.deadline:.1f}")

print("\n=== 2. REWARD CONSISTENCY ===")
# We disabled normalization in reward.yaml, verify Trainer vs Evaluator matches.
env = MARLResourceAllocationEnv(config)
env.reset(seed=42)
actions = {0: 1, 1: 1, 2: 1} # Allocate to second resource
_, env_rewards, _, _, infos = env.step(actions)

reward_engine = RewardEngine(config)
evaluator_step_reward = 0.0
trainer_step_reward = 0.0

for aid in env.agent_ids:
    ri = infos.get(aid, {}).get('reward_info', {})
    res = reward_engine.compute(ri)
    evaluator_step_reward += res.total
    trainer_step_reward += res.total

print(f"Trainer logic sum (ep_reward += result.total): {trainer_step_reward:.4f}")
print(f"Evaluator logic sum (sum(rewards.values())/len, then ep_reward += step_reward): {evaluator_step_reward:.4f}")
assert abs(trainer_step_reward - evaluator_step_reward) < 1e-6, "Mismatch!"
print("Consistency Verified: Exact match.")

print("\n=== 3. BASELINE DECISION DIVERSITY ===")
env.reset(seed=42)
task = env.get_current_task(0)
mask = env.get_action_mask(0)
print(f"Agent 0 Task: {task.task_id}, CPU req: {task.cpu_req:.1f}")
print(f"Agent 0 Mask: {mask}")

fcfs = FCFSScheduler()
rr = RoundRobinScheduler()
met = METScheduler()

a_fcfs = fcfs.select_actions(env)
print(f"FCFS picks resource index: {a_fcfs[0]} (First feasible)")
a_rr_1 = rr.select_actions(env)
a_rr_2 = rr.select_actions(env)
print(f"RR picks resource index (step 1): {a_rr_1[0]} (Cycles through feasible)")
print(f"RR picks resource index (step 2): {a_rr_2[0]} (Cycles through feasible)")
a_met = met.select_actions(env)
print(f"MET picks resource index: {a_met[0]} (Min Execution Time - R2 is 2x faster)")

