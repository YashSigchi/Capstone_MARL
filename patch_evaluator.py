import re

with open('marl/evaluation/evaluator.py', 'r') as f:
    content = f.read()

# Replace step_utilization extraction
old_util_block = """            # Utilization — all_utilizations is a list of per-resource utils
            all_utils = ri.get('all_utilizations', [])
            if all_utils and not step_utilization:
                # Only capture once per step (same across agents)
                step_utilization = [float(u) for u in all_utils]"""

new_util_block = """            # Utilization — all_utilizations is a list of per-resource utils
            all_utils = ri.get('all_utilizations', [])
            all_resources = ri.get('all_resources', [])
            if all_utils and not step_utilization:
                # Only capture once per step (same across agents)
                step_utilization = [float(u) for u in all_utils]
                # Also capture individual dimensions if available
                if all_resources:
                    step_cpu_utilization = [r.current_cpu_usage / max(r.cpu_capacity, 1e-6) for r in all_resources]
                    step_memory_utilization = [r.current_memory_usage / max(r.memory_capacity, 1e-6) for r in all_resources]"""

content = content.replace(old_util_block, new_util_block)

# Initialize new vars
content = content.replace("step_utilization = []", "step_utilization = []\n        step_cpu_utilization = []\n        step_memory_utilization = []")

# Pass to result
old_result_block = """        if step_utilization:
            result['utilization'] = step_utilization"""

new_result_block = """        if step_utilization:
            result['utilization'] = step_utilization
            result['cpu_utilization'] = step_cpu_utilization
            result['memory_utilization'] = step_memory_utilization"""

content = content.replace(old_result_block, new_result_block)

with open('marl/evaluation/evaluator.py', 'w') as f:
    f.write(content)
