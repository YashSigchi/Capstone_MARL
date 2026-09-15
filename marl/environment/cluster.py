"""
Status: Implemented
"""
from dataclasses import dataclass
from typing import List, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from marl.environment.resource import ResourceState
    from marl.environment.task import TaskRepresentation

@dataclass
class Cluster:
    """Cluster dataclass grouping resources under an agent."""
    cluster_id: str
    agent_id: int
    resource_ids: List[str]

    def get_resources(self, resource_registry: Dict[str, 'ResourceState']) -> List['ResourceState']:
        return [resource_registry[rid] for rid in self.resource_ids if rid in resource_registry]

    def aggregate_utilization(self, resource_registry: Dict[str, 'ResourceState']) -> float:
        resources = self.get_resources(resource_registry)
        if not resources:
            return 0.0
        return sum(r.utilization() for r in resources) / len(resources)

    def aggregate_available_capacity(self, resource_registry: Dict[str, 'ResourceState']) -> dict:
        resources = self.get_resources(resource_registry)
        return {
            'cpu': sum(r.available_cpu() for r in resources),
            'memory': sum(r.available_memory() for r in resources),
            'gpu': sum(r.available_gpu() for r in resources)
        }

    def get_feasible_resources(self, task: 'TaskRepresentation', resource_registry: Dict[str, 'ResourceState']) -> List['ResourceState']:
        return [r for r in self.get_resources(resource_registry) if r.can_execute(task)]

    @property
    def num_resources(self) -> int:
        return len(self.resource_ids)

    @classmethod
    def from_config(cls, config: dict) -> 'Cluster':
        """Create a Cluster from a config dict.

        Handles key mapping:
            'id' -> 'cluster_id'
            'resources' -> 'resource_ids'
        """
        mapped = dict(config)
        if 'id' in mapped and 'cluster_id' not in mapped:
            mapped['cluster_id'] = mapped.pop('id')
        if 'resources' in mapped and 'resource_ids' not in mapped:
            mapped['resource_ids'] = mapped.pop('resources')
        # agent_id defaults to -1, will be set by environment
        mapped.setdefault('agent_id', -1)
        import dataclasses
        known = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in mapped.items() if k in known}
        return cls(**filtered)
