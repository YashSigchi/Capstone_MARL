
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
if "def compute_action_penalty" not in content:
    content += action_penalty_code

with open('marl/reward/reward_components.py', 'w') as f:
    f.write(content)

# 2. Update configs/reward.yaml
with open('configs/reward.yaml', 'r') as f:
    yaml_lines = f.readlines()

new_yaml = []
inserted = False
for line in yaml_lines:
    new_yaml.append(line)
    if line.strip() == "components:":
        new_yaml.append("    action_penalty:\n      weight: 1.0\n")
        inserted = True

if not inserted:
    print("Could not find components: in reward.yaml")
else:
    with open('configs/reward.yaml', 'w') as f:
        f.writelines(new_yaml)

