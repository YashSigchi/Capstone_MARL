import re
with open('marl/reward/reward_components.py', 'r') as f:
    content = f.read()

content = re.sub(r"REWARD_COMPONENTS = \{", "REWARD_COMPONENTS = {\n    'action_penalty': {'fn': compute_action_penalty, 'scope': 'local'},", content)

with open('marl/reward/reward_components.py', 'w') as f:
    f.write(content)
