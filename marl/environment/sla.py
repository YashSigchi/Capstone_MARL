"""
Status: Implemented
"""
from dataclasses import dataclass
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from marl.environment.task import TaskRepresentation

@dataclass
class SLARecord:
    """SLA record dataclass."""
    task_id: str
    sla_class: str
    priority: int
    deadline: float
    completion_time: float
    violated: bool
    slack: float  # deadline - completion_time

class SLATracker:
    """SLA Tracker class."""
    
    def __init__(self):
        self.records: List[SLARecord] = []

    def check_violation(self, task: 'TaskRepresentation', completion_time: float) -> bool:
        return completion_time > task.deadline

    def compute_slack(self, task: 'TaskRepresentation', completion_time: float) -> float:
        return task.deadline - completion_time

    def record(self, task: 'TaskRepresentation', completion_time: float) -> SLARecord:
        violated = self.check_violation(task, completion_time)
        slack = self.compute_slack(task, completion_time)
        rec = SLARecord(
            task_id=task.task_id,
            sla_class=task.sla_class,
            priority=task.priority,
            deadline=task.deadline,
            completion_time=completion_time,
            violated=violated,
            slack=slack
        )
        self.records.append(rec)
        return rec

    def get_violation_rate(self) -> float:
        if not self.records:
            return 0.0
        violations = sum(1 for r in self.records if r.violated)
        return violations / len(self.records)

    def get_priority_violation_rate(self, min_priority: int = 3) -> float:
        relevant_records = [r for r in self.records if r.priority >= min_priority]
        if not relevant_records:
            return 0.0
        violations = sum(1 for r in relevant_records if r.violated)
        return violations / len(relevant_records)

    def get_class_violation_rate(self, sla_class: str) -> float:
        relevant_records = [r for r in self.records if r.sla_class == sla_class]
        if not relevant_records:
            return 0.0
        violations = sum(1 for r in relevant_records if r.violated)
        return violations / len(relevant_records)

    def get_average_slack(self) -> float:
        if not self.records:
            return 0.0
        return sum(r.slack for r in self.records) / len(self.records)

    def reset(self) -> None:
        self.records.clear()

    def get_summary(self) -> dict:
        return {
            'total_records': len(self.records),
            'violation_rate': self.get_violation_rate(),
            'high_priority_violation_rate': self.get_priority_violation_rate(),
            'average_slack': self.get_average_slack()
        }
