import yaml
from marl.environment.marl_env import MARLResourceAllocationEnv

with open('configs/environment.yaml', 'r') as f: env_cfg = yaml.safe_load(f)
with open('configs/marl.yaml', 'r') as f: env_cfg.update(yaml.safe_load(f))

env = MARLResourceAllocationEnv(env_cfg)
env.reset(seed=42)

for i in range(10):
    aid = list(env.agent_ids)[0]
    task = env.get_current_task(aid)
    cluster = env.clusters[env.agent_cluster_map[aid]]
    resources = cluster.get_resources(env.resources)
    print(f"\nStep {i} Task: {task.task_id} CPU req: {task.cpu_req:.2f}, Mem: {task.memory_req:.2f}, GPU: {task.gpu_req:.2f}")
    for r in resources:
        can = r.can_execute(task)
        print(f"  Res {r.resource_id} can_execute: {can} | CPU avail: {r.available_cpu():.2f}/{r.cpu_capacity:.2f} Mem avail: {r.available_memory():.2f}/{r.memory_capacity:.2f} GPU avail: {r.available_gpu():.2f}/{r.gpu_capacity:.2f}")
    
    # Try random allocate
    valid = [r for r in resources if r.can_execute(task)]
    if valid:
        act_idx = cluster.resource_ids.index(valid[0].resource_id)
        env.step({aid: act_idx})
    else:
        env.step({aid: cluster.num_resources + 1}) # REJECT
