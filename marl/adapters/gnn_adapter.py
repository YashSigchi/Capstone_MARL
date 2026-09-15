"""
GNN adapter utilities.

Status: Placeholder
"""
from abc import ABC, abstractmethod
from typing import Any, List
import numpy as np

# A minimal representation for typing purposes
class TaskRepresentation:
    def __init__(self, task_id: str, arrival_time: float, features: np.ndarray):
        self.task_id = task_id
        self.arrival_time = arrival_time
        self.features = features

class GNNAdapterInterface(ABC):
    """Interface for GNN encoder output → MARL task representations.
    
    Integration point for teammate's GNN encoder.
    The GNN produces task embeddings from the workflow graph.
    This adapter converts those embeddings into TaskRepresentations.
    
    Status: Placeholder
    """
    @abstractmethod
    def encode(self, task_graph: Any) -> List['TaskRepresentation']:
        """Encode task graph into TaskRepresentations."""
        pass
    
    @abstractmethod
    def get_embedding_dim(self) -> int:
        """Get the embedding dimensionality."""
        pass

class DummyGNNAdapter(GNNAdapterInterface):
    """Pass-through adapter that uses raw features as pseudo-embeddings.
    
    Placeholder until teammate's GNN is ready.
    
    Status: Placeholder
    """
    def __init__(self, embedding_dim: int = 16):
        """Initialize DummyGNNAdapter.
        
        Args:
            embedding_dim (int, optional): Dimension of embeddings. Defaults to 16.
        """
        self.embedding_dim = embedding_dim
    
    def encode(self, task_graph: Any) -> List['TaskRepresentation']:
        """Creates TaskRepresentations with random/feature-based embeddings.
        
        Args:
            task_graph (Any): Raw task graph.
            
        Returns:
            List['TaskRepresentation']: A list of encoded tasks.
        """
        dummy_feature = np.random.randn(self.embedding_dim).astype(np.float32)
        return [TaskRepresentation("dummy_task", 0.0, dummy_feature)]
    
    def get_embedding_dim(self) -> int:
        """Get the embedding dimensionality.
        
        Returns:
            int: Embedding dimension.
        """
        return self.embedding_dim
