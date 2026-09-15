import yaml
from marl.environment.marl_env import MARLResourceAllocationEnv
from marl.utils.seeding import set_all_seeds

with open('configs/environment.yaml', 'r') as f:
    env_cfg = yaml.safe_load(f)
with open('configs/marl.yaml', 'r') as f:
    marl_cfg = yaml.safe_load(f)

# Merge
config = dict(env_cfg)
config.update(marl_cfg)
config['max_steps'] = 3

def test_baselines_diversity():
    set_all_seeds(42)
    
    env = MARLResourceAllocationEnv(config)
    obs, _ = env.reset()
    
    print("Agent 0 (C1) resources:", env.clusters['C1'].resource_ids)
    for rid in env.clusters['C1'].resource_ids:
        r = env.resources[rid]
        print(f"  {rid}: CPU={r.cpu_capacity}, GPU={r.gpu_capacity}")
    
    task = env.get_current_task(0)
    if task: print(f"Current Task for Agent 0: CPU req={task.cpu_req:.2f}, GPU req={task.gpu_req:.2f}")
    
    mask = env.get_action_mask(0)
    print(f"Action Mask for Agent 0: {mask}")
