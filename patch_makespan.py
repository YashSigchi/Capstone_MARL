with open('marl/environment/marl_env.py', 'r') as f:
    content = f.read()

# Replace the makespan reporting in the step method
old_step_info_makespan = """            info["total_tasks_processed"] = (
                len(self._completed_tasks) + len(self._rejected_tasks)
            )
            info["makespan"] = self._makespan
            info["unfinished_tasks"] = sum(q.length for q in self.queues.values()) + len(self._running_tasks)"""

new_step_info_makespan = """            info["total_tasks_processed"] = (
                len(self._completed_tasks) + len(self._rejected_tasks)
            )
            
            # True makespan semantics: if there are unfinished tasks, the true makespan 
            # is at least the current step. If all are done, it's the exact time the last task finished.
            unfinished = sum(q.length for q in self.queues.values()) + len(self._running_tasks)
            info["unfinished_tasks"] = unfinished
            info["makespan"] = self._makespan if unfinished == 0 else float(self._current_step)
"""

content = content.replace(old_step_info_makespan, new_step_info_makespan)

# And in default step info
old_default = """            "makespan": getattr(self, "_makespan", 0.0),"""
new_default = """            "makespan": getattr(self, "_makespan", 0.0) if (sum(q.length for q in self.queues.values()) + len(self._running_tasks)) == 0 else float(self._current_step),"""

content = content.replace(old_default, new_default)

with open('marl/environment/marl_env.py', 'w') as f:
    f.write(content)
