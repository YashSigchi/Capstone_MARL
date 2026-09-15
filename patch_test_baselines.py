import re
with open('tests/test_baselines.py', 'r') as f:
    content = f.read()

content = content.replace("sjf", "met")
content = content.replace("SJFScheduler", "METScheduler")
content = content.replace("sjf_picks_fastest", "met_picks_fastest")

with open('tests/test_baselines.py', 'w') as f:
    f.write(content)
