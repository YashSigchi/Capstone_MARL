"""
Status: Implemented
"""
import numpy as np
from typing import List, Optional

class NetworkTopology:
    """NetworkTopology managing connectivity between resources/clusters."""
    def __init__(self, adjacency_matrix: np.ndarray, node_ids: List[str],
                 bandwidth_matrix: Optional[np.ndarray] = None,
                 latency_matrix: Optional[np.ndarray] = None):
        self.adjacency_matrix = adjacency_matrix
        self.node_ids = node_ids
        self._node_to_idx = {nid: i for i, nid in enumerate(node_ids)}
        
        n = len(node_ids)
        self.bandwidth_matrix = bandwidth_matrix if bandwidth_matrix is not None else np.full((n, n), 10.0) * adjacency_matrix
        self.latency_matrix = latency_matrix if latency_matrix is not None else np.full((n, n), 1.0) * adjacency_matrix

    def get_neighbors(self, node_id: str) -> List[str]:
        idx = self._node_to_idx[node_id]
        return [self.node_ids[i] for i, connected in enumerate(self.adjacency_matrix[idx]) if connected > 0]

    def get_bandwidth(self, src: str, dst: str) -> float:
        src_idx, dst_idx = self._node_to_idx[src], self._node_to_idx[dst]
        return self.bandwidth_matrix[src_idx, dst_idx]

    def get_latency(self, src: str, dst: str) -> float:
        src_idx, dst_idx = self._node_to_idx[src], self._node_to_idx[dst]
        return self.latency_matrix[src_idx, dst_idx]

    def are_connected(self, src: str, dst: str) -> bool:
        src_idx, dst_idx = self._node_to_idx[src], self._node_to_idx[dst]
        return self.adjacency_matrix[src_idx, dst_idx] > 0

    @property
    def num_nodes(self) -> int:
        return len(self.node_ids)

    def to_feature_vector(self) -> np.ndarray:
        return self.adjacency_matrix.flatten()

    @classmethod
    def from_type(cls, topology_type: str, node_ids: List[str], **kwargs) -> 'NetworkTopology':
        n = len(node_ids)
        adj = np.zeros((n, n))
        
        if topology_type == 'line':
            for i in range(n - 1):
                adj[i, i + 1] = 1
                adj[i + 1, i] = 1
        elif topology_type == 'fully_connected':
            adj = np.ones((n, n)) - np.eye(n)
        elif topology_type == 'grid':
            side = int(np.ceil(np.sqrt(n)))
            for i in range(n):
                r, c = divmod(i, side)
                if r > 0: adj[i, i - side] = 1
                if r < side - 1 and i + side < n: adj[i, i + side] = 1
                if c > 0: adj[i, i - 1] = 1
                if c < side - 1 and i + 1 < n: adj[i, i + 1] = 1
        elif topology_type == 'random':
            p = kwargs.get('edge_probability', 0.5)
            adj = (np.random.rand(n, n) < p).astype(float)
            adj = np.tril(adj, -1) + np.tril(adj, -1).T
        else:
            raise ValueError(f"Unknown topology type: {topology_type}")
            
        np.fill_diagonal(adj, 1) # nodes connected to themselves
            
        return cls(adjacency_matrix=adj, node_ids=node_ids)
