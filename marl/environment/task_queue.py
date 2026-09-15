"""
Status: Implemented
"""
from typing import List, Optional
from marl.environment.task import TaskRepresentation

class TaskQueue:
    """Environment-owned task queue."""
    def __init__(self, ordering: str = 'fifo'):
        self.ordering = ordering
        self._tasks: List[TaskRepresentation] = []

    def _sort_tasks(self) -> None:
        if self.ordering == 'priority':
            # Higher priority first
            self._tasks.sort(key=lambda t: t.priority, reverse=True)
        elif self.ordering == 'deadline':
            # Earlier deadline first
            self._tasks.sort(key=lambda t: t.deadline)

    def enqueue(self, task: TaskRepresentation) -> None:
        self._tasks.append(task)
        if self.ordering != 'fifo':
            self._sort_tasks()

    def dequeue(self) -> Optional[TaskRepresentation]:
        if not self._tasks:
            return None
        return self._tasks.pop(0)

    def peek(self) -> Optional[TaskRepresentation]:
        if not self._tasks:
            return None
        return self._tasks[0]

    def remove(self, task_id: str) -> Optional[TaskRepresentation]:
        for i, task in enumerate(self._tasks):
            if task.task_id == task_id:
                return self._tasks.pop(i)
        return None

    def contains(self, task_id: str) -> bool:
        return any(task.task_id == task_id for task in self._tasks)

    @property
    def length(self) -> int:
        return len(self._tasks)

    @property
    def is_empty(self) -> bool:
        return len(self._tasks) == 0

    def get_waiting_times(self, current_time: float) -> List[float]:
        return [task.get_waiting_time(current_time) for task in self._tasks]

    def average_waiting_time(self, current_time: float) -> float:
        if not self._tasks:
            return 0.0
        return sum(self.get_waiting_times(current_time)) / len(self._tasks)

    def get_all_tasks(self) -> List[TaskRepresentation]:
        return list(self._tasks)

    def clear(self) -> None:
        self._tasks.clear()
