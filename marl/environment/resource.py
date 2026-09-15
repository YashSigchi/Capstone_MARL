"""
Status: Implemented
"""
from dataclasses import dataclass
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from marl.environment.task import TaskRepresentation

@dataclass
class ResourceState:
    """ResourceState dataclass representing a heterogeneous compute resource."""
    resource_id: str
    cpu_capacity: float
    memory_capacity: float
    gpu_capacity: float
    network_bandwidth: float
    power_rate: float          # Watts at full load
    processing_speed: float    # relative speed multiplier (1.0 = baseline)
    active: bool = True
    # Dynamic state
    current_cpu_usage: float = 0.0
    current_memory_usage: float = 0.0
    current_gpu_usage: float = 0.0

    def available_cpu(self) -> float:
        return max(0.0, self.cpu_capacity - self.current_cpu_usage)

    def available_memory(self) -> float:
        return max(0.0, self.memory_capacity - self.current_memory_usage)

    def available_gpu(self) -> float:
        return max(0.0, self.gpu_capacity - self.current_gpu_usage)

    def utilization(self) -> float:
        utils = []
        if self.cpu_capacity > 0:
            utils.append(self.current_cpu_usage / self.cpu_capacity)
        if self.memory_capacity > 0:
            utils.append(self.current_memory_usage / self.memory_capacity)
        if self.gpu_capacity > 0:
            utils.append(self.current_gpu_usage / self.gpu_capacity)
        return sum(utils) / len(utils) if utils else 0.0

    def can_execute(self, task: 'TaskRepresentation') -> bool:
        if not self.active:
            return False
        return (self.available_cpu() >= task.cpu_req and
                self.available_memory() >= task.memory_req and
                self.available_gpu() >= task.gpu_req)

    def allocate(self, task: 'TaskRepresentation') -> None:
        if not self.can_execute(task):
            raise ValueError(f"Insufficient capacity on resource {self.resource_id} for task {task.task_id}")
        self.current_cpu_usage += task.cpu_req
        self.current_memory_usage += task.memory_req
        self.current_gpu_usage += task.gpu_req

    def release(self, task: 'TaskRepresentation') -> None:
        self.current_cpu_usage = max(0.0, self.current_cpu_usage - task.cpu_req)
        self.current_memory_usage = max(0.0, self.current_memory_usage - task.memory_req)
        self.current_gpu_usage = max(0.0, self.current_gpu_usage - task.gpu_req)

    def reset(self) -> None:
        self.current_cpu_usage = 0.0
        self.current_memory_usage = 0.0
        self.current_gpu_usage = 0.0
        self.active = True

    def to_feature_vector(self) -> np.ndarray:
        return np.array([
            self.cpu_capacity,
            self.memory_capacity,
            self.gpu_capacity,
            self.network_bandwidth,
            self.power_rate,
            self.processing_speed,
            float(self.active),
            self.current_cpu_usage,
            self.current_memory_usage,
            self.current_gpu_usage
        ], dtype=np.float32)

    @classmethod
    def from_config(cls, config: dict) -> 'ResourceState':
        """Create a ResourceState from a config dict.

        Handles key mapping: 'id' -> 'resource_id'.
        Ignores unknown keys.
        """
        mapped = dict(config)
        # Map 'id' to 'resource_id'
        if 'id' in mapped and 'resource_id' not in mapped:
            mapped['resource_id'] = mapped.pop('id')
        # Only pass known fields
        import dataclasses
        known = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in mapped.items() if k in known}
        return cls(**filtered)
