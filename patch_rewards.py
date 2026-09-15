import yaml

# 1. Update reward_components.py
with open('marl/reward/reward_components.py', 'r') as f:
    content = f.read()

# Remove reject/defer from latency
old_latency = """    if env_info.get('action_was_reject', False):
        return -1.0
    if env_info.get('action_was_defer', False):
        return -0.1
    
    time = env_info.get('total_completion_time', 0.0)"""

new_latency = """    time = env_info.get('total_completion_time', 0.0)"""
content = content.replace(old_latency, new_latency)

# Remove reject/defer from SLA
old_sla = """    if env_info.get('action_was_reject', False):
        return -1.0
    if env_info.get('action_was_defer', False):
        return -0.05
        
    if not env_info.get('sla_violated', False):"""

new_sla = """    if not env_info.get('sla_violated', False):"""
content = content.replace(old_sla, new_sla)

# Add compute_action_penalty
action_penalty_code = """
def compute_action_penalty(env_info: dict) -> float:
    \"\"\"
    Applies explicit penalties for agent's local scheduling actions (DEFER/REJECT).
    [LOCAL]
    \"\"\"
    if env_info.get('action_was_reject', False):
        return -1.0
    if env_info.get('action_was_defer', False):
        return -0.1
    return 0.0
"""
content += action_penalty_code

with open('marl/reward/reward_components.py', 'w') as f:
    f.write(content)

# 2. Update configs/reward.yaml
with open('configs/reward.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Add action_penalty to reward components
if 'action_penalty' not in config['reward']['components']:
    config['reward']['components']['action_penalty'] = {'weight': 1.0}

with open('configs/reward.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)

