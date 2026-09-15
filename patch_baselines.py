import os
import glob

for filename in glob.glob('marl/baselines/*.py'):
    with open(filename, 'r') as f:
        content = f.read()
    
    # Change len(mask) - 1 to len(mask) - 2 for FCFS and SJF etc where they avoid defer
    content = content.replace("for i in range(len(mask) - 1):", "for i in range(len(mask) - 2):")
    content = content.replace("action = len(mask) - 1", "action = len(mask) - 2")
    
    with open(filename, 'w') as f:
        f.write(content)
