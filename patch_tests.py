import re

with open('tests/test_action_mask.py', 'r') as f:
    content = f.read()

content = content.replace("mask[:-1]", "mask[:-2]")
content = content.replace("mask[-1] == 1.0", "mask[-1] == 1.0 and mask[-2] == 1.0")

with open('tests/test_action_mask.py', 'w') as f:
    f.write(content)


with open('tests/test_environment.py', 'r') as f:
    content = f.read()

content = content.replace("env.action_space(0).n == 3", "env.action_space(0).n == 4")

with open('tests/test_environment.py', 'w') as f:
    f.write(content)
