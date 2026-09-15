"""
Status: Implemented
"""
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from marl.environment.task import TaskRepresentation
    from marl.environment.resource import ResourceState
    from marl.environment.network import NetworkTopology


class ExecutionModel(ABC):
    """Decomposed latency model."""
    
    @abstractmethod
    def compute_execution_time(self, task: 'TaskRepresentation', resource: 'ResourceState') -> float:
        pass
        
    @abstractmethod
    def compute_network_time(self, task: 'TaskRepresentation', src_id: str, dst_id: str, topology: 'NetworkTopology') -> float:
        pass

    def compute_queue_waiting_time(self, task: 'TaskRepresentation', current_time: float) -> float:
        return current_time - task.arrival_time

    def compute_total_time(self, task: 'TaskRepresentation', resource: 'ResourceState', current_time: float, src_id: str, dst_id: str, topology: 'NetworkTopology') -> float:
        return (self.compute_queue_waiting_time(task, current_time)
                + self.compute_network_time(task, src_id, dst_id, topology)
                + self.compute_execution_time(task, resource))


class SimpleExecutionModel(ExecutionModel):
    """Simple execution model implementation."""
    
    def compute_execution_time(self, task: 'TaskRepresentation', resource: 'ResourceState') -> float:
        return task.work_units / resource.processing_speed
        
    def compute_network_time(self, task: 'TaskRepresentation', src_id: str, dst_id: str, topology: 'NetworkTopology') -> float:
        if src_id == dst_id:
            return 0.0
        latency = topology.get_latency(src_id, dst_id)
        bandwidth = topology.get_bandwidth(src_id, dst_id)
        transfer_time = task.data_size / max(bandwidth, 1e-6)
        return latency + transfer_time
