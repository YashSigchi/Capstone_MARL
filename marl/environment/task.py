"""
Status: Implemented
"""
from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from marl.environment.resource import ResourceState

@dataclass
class TaskRepresentation:
    """TaskRepresentation dataclass."""
    task_id: str
    cpu_req: float
    memory_req: float
    gpu_req: float
    work_units: float          # explicit computational workload
    deadline: float            # absolute timestep deadline
    priority: int              # 1=lowest, 5=highest
    data_size: float           # MB
    communication_req: float   # communication overhead factor
    dependencies: List[str]    # task_ids this depends on
    arrival_time: float        # timestep when task arrived
    sla_class: str             # 'gold', 'silver', 'bronze'
    embedding: Optional[np.ndarray] = None  # GNN embedding slot
    # Tracking fields set by environment
    assigned_resource_id: Optional[str] = None
    start_time: Optional[float] = None
    completion_time: Optional[float] = None
    status: str = 'pending'    # pending, queued, running, completed, rejected

    def to_feature_vector(self) -> np.ndarray:
        return np.array([
            self.cpu_req,
            self.memory_req,
            self.gpu_req,
            self.work_units,
            self.deadline,
            float(self.priority),
            self.data_size,
            self.communication_req,
            self.arrival_time
        ], dtype=np.float32)

    def is_feasible_on(self, resource: 'ResourceState') -> bool:
        return resource.can_execute(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'TaskRepresentation':
        return cls(**d)

    def get_waiting_time(self, current_time: float) -> float:
        return max(0.0, current_time - self.arrival_time)


class TaskGenerator:
    """Generator for creating random tasks."""
    def __init__(self, config: dict, rng: np.random.Generator):
        self.config = config
        self.rng = rng

    def _get_range(self, key: str, default: tuple) -> tuple:
        """Extract (low, high) from config, handling both dict and tuple formats."""
        val = self.config.get(key, default)
        if isinstance(val, dict):
            return (val.get('min', default[0]), val.get('max', default[1]))
        return tuple(val)

    def generate_task(self, task_id: str, arrival_time: float) -> TaskRepresentation:
        cpu_range = self._get_range('cpu_req', (0.1, 1.0))
        mem_range = self._get_range('memory_req', (0.1, 1.0))
        gpu_range = self._get_range('gpu_req', (0.0, 1.0))
        wu_range = self._get_range('work_units', (10.0, 100.0))
        dl_range = self._get_range('deadline', (10.0, 50.0))
        prio_range = self._get_range('priority', (1, 5))
        ds_range = self._get_range('data_size', (1.0, 100.0))
        comm_range = self._get_range('communication_req', (0.0, 1.0))

        sla_classes = self.config.get('sla_classes', ['bronze', 'silver', 'gold'])
        sla_weights = self.config.get('sla_class_weights', None)
        if sla_weights is not None:
            sla_weights_arr = np.array(sla_weights, dtype=float)
            sla_weights_arr = sla_weights_arr / sla_weights_arr.sum()
        else:
            sla_weights_arr = None

        return TaskRepresentation(
            task_id=task_id,
            cpu_req=float(self.rng.uniform(cpu_range[0], cpu_range[1])),
            memory_req=float(self.rng.uniform(mem_range[0], mem_range[1])),
            gpu_req=float(self.rng.uniform(gpu_range[0], gpu_range[1])),
            work_units=float(self.rng.uniform(wu_range[0], wu_range[1])),
            deadline=arrival_time + float(self.rng.uniform(dl_range[0], dl_range[1])),
            priority=int(self.rng.integers(int(prio_range[0]), int(prio_range[1]) + 1)),
            data_size=float(self.rng.uniform(ds_range[0], ds_range[1])),
            communication_req=float(self.rng.uniform(comm_range[0], comm_range[1])),
            dependencies=[],
            arrival_time=arrival_time,
            sla_class=str(self.rng.choice(sla_classes, p=sla_weights_arr)),
        )

    def generate_batch(self, count: int, arrival_time: float) -> List[TaskRepresentation]:
        return [self.generate_task(f"task_{arrival_time}_{i}", arrival_time) for i in range(count)]
