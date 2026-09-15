import os

# Patch met.py
with open('marl/baselines/met.py', 'r') as f:
    content = f.read()

content = content.replace("SJFScheduler", "METScheduler")
content = content.replace("Shortest-Job-First:", "Minimum Execution Time:")
content = content.replace("name='sjf'", "name='met'")

with open('marl/baselines/met.py', 'w') as f:
    f.write(content)


# Patch test_met_deterministic.py
with open('tests/test_met_deterministic.py', 'r') as f:
    content = f.read()

content = content.replace("SJFScheduler", "METScheduler")
content = content.replace("sjf_vs_fcfs", "met_vs_fcfs")
content = content.replace("sjf = ", "met = ")
content = content.replace("actions = sjf.select_actions", "actions = met.select_actions")
content = content.replace("SJF should pick", "MET should pick")
content = content.replace("marl.baselines.sjf", "marl.baselines.met")

with open('tests/test_met_deterministic.py', 'w') as f:
    f.write(content)

# Patch scripts/run_baselines.py
with open('scripts/run_baselines.py', 'r') as f:
    content = f.read()

content = content.replace("sjf", "met")
content = content.replace("SJFScheduler", "METScheduler")
content = content.replace("marl.baselines.met", "marl.baselines.met") # Already handled by replace

with open('scripts/run_baselines.py', 'w') as f:
    f.write(content)

