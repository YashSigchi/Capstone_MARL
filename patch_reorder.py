with open('marl/reward/reward_components.py', 'r') as f:
    content = f.read()

import re
# Extract the dictionary
dict_match = re.search(r"REWARD_COMPONENTS = \{.*?\n\}", content, re.DOTALL)
if dict_match:
    dict_str = dict_match.group(0)
    # Remove it from the current position
    content = content.replace(dict_str, "")
    # Add it to the end
    content += "\n\n" + dict_str

with open('marl/reward/reward_components.py', 'w') as f:
    f.write(content)
