import re
import os

filepath = 'marl/environment/marl_env.py'
with open(filepath, 'r') as f:
    content = f.read()

# 1. Action space
content = content.replace(
    'return spaces.Discrete(n_resources + 1)  # +1 for defer',
    'return spaces.Discrete(n_resources + 2)  # +1 for defer, +1 for reject'
)

# 2. _get_defer_action and _get_reject_action
content = content.replace(
    'def _get_defer_action(self, agent_id: int) -> int:\n        """Return the defer action index for the agent."""\n        cluster = self._get_cluster(agent_id)\n        return cluster.num_resources  # last action',
    'def _get_defer_action(self, agent_id: int) -> int:\n        """Return the defer action index for the agent."""\n        cluster = self._get_cluster(agent_id)\n        return cluster.num_resources\n\n    def _get_reject_action(self, agent_id: int) -> int:\n        """Return the reject action index for the agent."""\n        cluster = self._get_cluster(agent_id)\n        return cluster.num_resources + 1'
)

# 3. get_action_mask
old_mask = """        n_actions = len(resources) + 1  # +1 for defer

        mask = np.zeros(n_actions, dtype=np.float32)
        task = self.get_current_task(agent_id)

        if task is not None:
            for i, r in enumerate(resources):
                if r.can_execute(task):
                    mask[i] = 1.0
        # Defer is always valid
        mask[-1] = 1.0"""
new_mask = """        n_actions = len(resources) + 2  # +1 for defer, +1 for reject

        mask = np.zeros(n_actions, dtype=np.float32)
        task = self.get_current_task(agent_id)

        if task is not None:
            for i, r in enumerate(resources):
                if r.can_execute(task):
                    mask[i] = 1.0
        # Defer and Reject are always valid
        mask[-2] = 1.0
        mask[-1] = 1.0"""
content = content.replace(old_mask, new_mask)

# 4. get_action_dim
content = content.replace(
    'return cluster.num_resources + 1',
    'return cluster.num_resources + 2'
)

# 5. reset makespan
content = content.replace(
    'self._total_carbon = 0.0\n        self.sla_tracker.reset()',
    'self._total_carbon = 0.0\n        self._makespan = 0.0\n        self.sla_tracker.reset()'
)

# 6. _process_action
old_process = """        defer_action = len(resources)
        task = self.queues[agent_id].peek()

        info: Dict[str, Any] = {
            "agent_id": agent_id,
            "action_was_defer": True,
            "allocated_task": None,
            "allocated_resource": None,
            "execution_time": 0.0,
            "network_time": 0.0,
            "queue_waiting_time": 0.0,
            "total_completion_time": 0.0,
            "energy_consumed": 0.0,
            "carbon_emitted": 0.0,
            "sla_violated": False,
            "sla_slack": 0.0,
            "task_priority": 0,
            "local_queue_length": self.queues[agent_id].length,
            "local_avg_waiting_time": self.queues[agent_id].average_waiting_time(
                float(self._current_step)
            ),
            "local_resources": resources,
        }

        if task is None or action == defer_action:
            # No task or agent chose to defer
            self._step_info[agent_id] = info
            return

        # Validate action
        if action < 0 or action >= len(resources):
            # Invalid action index → treat as defer
            self._step_info[agent_id] = info
            return

        resource = resources[action]

        if not resource.can_execute(task):
            # Resource cannot handle this task → treat as defer
            self._step_info[agent_id] = info
            return"""

new_process = """        defer_action = len(resources)
        reject_action = len(resources) + 1
        task = self.queues[agent_id].peek()

        info: Dict[str, Any] = {
            "agent_id": agent_id,
            "action_was_defer": True,
            "action_was_reject": False,
            "allocated_task": None,
            "allocated_resource": None,
            "execution_time": 0.0,
            "network_time": 0.0,
            "queue_waiting_time": 0.0,
            "total_completion_time": 0.0,
            "energy_consumed": 0.0,
            "carbon_emitted": 0.0,
            "sla_violated": False,
            "sla_slack": 0.0,
            "task_priority": 0,
            "local_queue_length": self.queues[agent_id].length,
            "local_avg_waiting_time": self.queues[agent_id].average_waiting_time(
                float(self._current_step)
            ),
            "local_resources": resources,
        }

        if task is None or action == defer_action:
            # No task or agent chose to defer
            self._step_info[agent_id] = info
            return
            
        if action == reject_action:
            task = self.queues[agent_id].dequeue()
            task.status = "rejected"
            self._rejected_tasks.append(task)
            info["action_was_defer"] = False
            info["action_was_reject"] = True
            self._step_info[agent_id] = info
            return

        # Validate action
        if action < 0 or action >= len(resources):
            # Invalid action index → treat as defer
            self._step_info[agent_id] = info
            return

        resource = resources[action]

        if not resource.can_execute(task):
            # Resource cannot handle this task → treat as defer
            self._step_info[agent_id] = info
            return"""
content = content.replace(old_process, new_process)

# 7. makespan update
content = content.replace(
    'self._completed_tasks.append(rt.task)',
    'self._completed_tasks.append(rt.task)\n                self._makespan = max(self._makespan, completion_time)'
)

# 8. Add global info
content = content.replace(
    '"total_tasks_processed": (\n                len(self._completed_tasks) + len(self._rejected_tasks)\n            ),',
    '"total_tasks_processed": (\n                len(self._completed_tasks) + len(self._rejected_tasks)\n            ),\n            "makespan": self._makespan,\n            "unfinished_tasks": sum(q.length for q in self.queues.values()) + len(self._running_tasks),'
)

content = content.replace(
    '"total_tasks_processed": (\n                len(self._completed_tasks) + len(self._rejected_tasks)\n            )\n        }',
    '"total_tasks_processed": (\n                len(self._completed_tasks) + len(self._rejected_tasks)\n            ),\n            "makespan": getattr(self, "_makespan", 0.0),\n            "unfinished_tasks": sum(q.length for q in self.queues.values()) + len(self._running_tasks)\n        }'
)

with open(filepath, 'w') as f:
    f.write(content)
print("marl_env.py patched")
