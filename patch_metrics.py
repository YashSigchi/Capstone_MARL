import re

with open('marl/evaluation/metrics.py', 'r') as f:
    content = f.read()

# Add tracking lists
content = content.replace("self._current_utilizations: List[List[float]] = []", 
                          "self._current_utilizations: List[List[float]] = []\n        self._current_cpu_utilizations: List[List[float]] = []\n        self._current_memory_utilizations: List[List[float]] = []")

# Append data
old_append = "if 'utilization' in step_info:\n            self._current_utilizations.append(step_info['utilization'])"
new_append = """if 'utilization' in step_info:
            self._current_utilizations.append(step_info['utilization'])
        if 'cpu_utilization' in step_info:
            self._current_cpu_utilizations.append(step_info['cpu_utilization'])
        if 'memory_utilization' in step_info:
            self._current_memory_utilizations.append(step_info['memory_utilization'])"""
content = content.replace(old_append, new_append)

# Compute metrics
old_compute = """        if self._current_utilizations:
            utils = np.array(self._current_utilizations)
            metrics.avg_cpu_utilization = float(np.mean(utils))
            # simplified for now, assuming scalar util, can be expanded
            metrics.avg_memory_utilization = float(np.mean(utils))
            metrics.resource_imbalance = float(np.std(utils))"""
new_compute = """        if self._current_utilizations:
            utils = np.array(self._current_utilizations)
            metrics.resource_imbalance = float(np.std(utils))
        if self._current_cpu_utilizations:
            cpu_utils = np.array(self._current_cpu_utilizations)
            metrics.avg_cpu_utilization = float(np.mean(cpu_utils))
        if self._current_memory_utilizations:
            mem_utils = np.array(self._current_memory_utilizations)
            metrics.avg_memory_utilization = float(np.mean(mem_utils))"""
content = content.replace(old_compute, new_compute)

# Clear states
old_clear = "self._current_utilizations.clear()"
new_clear = "self._current_utilizations.clear()\n        self._current_cpu_utilizations.clear()\n        self._current_memory_utilizations.clear()"
content = content.replace(old_clear, new_clear)

with open('marl/evaluation/metrics.py', 'w') as f:
    f.write(content)
