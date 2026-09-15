import yaml
from marl.environment.marl_env import MARLResourceAllocationEnv
from marl.reward.reward_engine import RewardEngine
from marl.baselines.fcfs import FCFSScheduler

with open('configs/environment.yaml', 'r') as f:
    env_cfg = yaml.safe_load(f)
with open('configs/marl.yaml', 'r') as f:
    marl_cfg = yaml.safe_load(f)

config = dict(env_cfg)
config.update(marl_cfg)
config['max_steps'] = 20
env = MARLResourceAllocationEnv(config)
reward_engine = RewardEngine(config)

obs, _ = env.reset(seed=42)
fcfs = FCFSScheduler()

total_r = 0.0
for step in range(20):
    actions = fcfs.select_actions(env)
    obs, _, _, _, infos = env.step(actions)
    for aid in env.agent_ids:
        ri = infos[aid].get('reward_info', {})
        res = reward_engine.compute(ri)
        total_r += res.total
        print(f"Step {step}, Agent {aid} Total: {res.total:.4f}, Components: {res.components}")
print(f"Final Total: {total_r}")
