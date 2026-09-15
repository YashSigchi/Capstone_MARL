"""
Workload adapter utilities.

Status: Placeholder (interface) + Implemented (simple encoder)
"""
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Dict
import numpy as np

class TaskRepresentation:
    """Representation of a task.
    
    Status: Implemented
    """
    def __init__(self, task_id: str, arrival_time: float, features: np.ndarray):
        """Initialize the TaskRepresentation.
        
        Args:
            task_id (str): ID of the task.
            arrival_time (float): Task arrival time.
            features (np.ndarray): Array of features.
        """
        self.task_id = task_id
        self.arrival_time = arrival_time
        self.features = features

class WorkloadEncoderInterface(ABC):
    """Interface for encoding raw workloads into TaskRepresentations.
    
    Integration point for teammate's workload processor.
    
    Status: Placeholder (interface)
    """
    @abstractmethod
    def encode(self, workload: Any) -> List['TaskRepresentation']:
        """Encode raw workload into a list of TaskRepresentations."""
        pass

class SimpleWorkloadEncoder(WorkloadEncoderInterface):
    """Converts raw task dicts/lists into TaskRepresentations.
    
    This is the default encoder that transforms structured task data
    into TaskRepresentation objects with numerical feature vectors.
    
    Status: Implemented
    """
    def __init__(self, config: Optional[dict] = None):
        """Initialize the SimpleWorkloadEncoder.
        
        Args:
            config (Optional[dict], optional): Configuration dictionary. Defaults to None.
        """
        self.config = config or {}
        self.default_features = self.config.get('default_features', [0.0, 0.0, 0.0])
    
    def encode(self, workload: Any) -> List['TaskRepresentation']:
        """Accept: List[dict] or dict with 'tasks' key.
        Each dict should have task fields.
        Missing fields get reasonable defaults.
        
        Args:
            workload (Any): Raw workload data.
            
        Returns:
            List['TaskRepresentation']: A list of encoded tasks.
        """
        tasks_data = []
        if isinstance(workload, dict) and 'tasks' in workload:
            tasks_data = workload['tasks']
        elif isinstance(workload, list):
            tasks_data = workload
            
        encoded_tasks = []
        for i, task_dict in enumerate(tasks_data):
            task_id = task_dict.get('id', f"task_{i}")
            arrival_time = task_dict.get('arrival_time', 0.0)
            encoded_tasks.append(self._dict_to_task(task_dict, task_id, arrival_time))
            
        return encoded_tasks
    
    def _dict_to_task(self, d: dict, task_id: str, arrival_time: float = 0.0) -> 'TaskRepresentation':
        """Convert a single dictionary to a TaskRepresentation.
        
        Args:
            d (dict): Task dictionary.
            task_id (str): Task ID.
            arrival_time (float, optional): Task arrival time. Defaults to 0.0.
            
        Returns:
            TaskRepresentation: The task representation.
        """
        features = d.get('features', self.default_features)
        return TaskRepresentation(
            task_id=task_id,
            arrival_time=arrival_time,
            features=np.array(features, dtype=np.float32)
        )
