with open('marl/reward/reward_components.py', 'r') as f:
    content = f.read()

import re

# Remove the action_was_reject and action_was_defer from compute_latency_reward
content = re.sub(r"    if env_info.get\('action_was_reject', False\):\n        return -1.0\n    if env_info.get\('action_was_defer', False\):\n        return -0.1\n\n", "", content)

content = re.sub(r"    if env_info.get\('action_was_reject', False\):\n        return -1.0\n    if env_info.get\('action_was_defer', False\):\n        return -0.05\n\n", "", content)

# Check if action_penalty exists
if "def compute_action_penalty" not in content:
    content += """
def compute_action_penalty(env_info: Dict[str, Any]) -> float:
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

with open('marl/reward/reward_components.py', 'w') as f:
    f.write(content)
