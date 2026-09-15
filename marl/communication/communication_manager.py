"""
Communication manager utilities.

Status: Experimental
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import numpy as np

class NetworkTopology:
    """Mock network topology class for typing purposes."""
    def get_neighbors(self, agent_id: int) -> List[int]:
        return []

@dataclass
class AgentMessage:
    """Message from one agent to another."""
    sender_id: int
    receiver_id: int
    content: Dict[str, float]  # e.g., queue_length, utilization, available_cpu
    timestamp: int

class CommunicationManager:
    """Manages optional inter-agent communication.
    
    When enabled, agents can share selected local information
    with their neighbors (defined by network topology).
    
    Messages contain: queue length, utilization, available capacity.
    No unrestricted full-state broadcasting.
    
    Status: Experimental
    """
    def __init__(self, enabled: bool = False, 
                 topology: Optional['NetworkTopology'] = None,
                 message_fields: Optional[List[str]] = None):
        """Initialize CommunicationManager.
        
        Args:
            enabled (bool, optional): Whether communication is enabled. Defaults to False.
            topology (Optional[NetworkTopology], optional): The network topology. Defaults to None.
            message_fields (Optional[List[str]], optional): Fields to include in messages. Defaults to None.
        """
        self.enabled = enabled
        self.topology = topology
        self.message_fields = message_fields or ['queue_length', 'utilization', 'available_cpu']
        
        self.num_messages_sent = 0
        self.total_data_size = 0.0
    
    def collect_messages(self, agent_id: int, 
                         all_agent_states: Dict[int, dict]) -> List[AgentMessage]:
        """Collect messages from neighboring agents.
        
        Args:
            agent_id (int): The ID of the receiving agent.
            all_agent_states (Dict[int, dict]): States of all agents.
            
        Returns:
            List[AgentMessage]: A list of incoming messages.
        """
        if not self.enabled or self.topology is None:
            return []
            
        neighbors = self.topology.get_neighbors(agent_id)
        messages = []
        current_time = 0  # Assuming timestamp is fetched from environment in a real implementation
        
        for neighbor_id in neighbors:
            if neighbor_id in all_agent_states:
                state = all_agent_states[neighbor_id]
                content = {field: float(state.get(field, 0.0)) for field in self.message_fields}
                msg = AgentMessage(
                    sender_id=neighbor_id,
                    receiver_id=agent_id,
                    content=content,
                    timestamp=current_time
                )
                messages.append(msg)
                
                self.num_messages_sent += 1
                self.total_data_size += len(self.message_fields) * 4  # Assuming 4 bytes per float
                
        return messages
    
    def encode_messages(self, messages: List[AgentMessage]) -> np.ndarray:
        """Encode received messages into a fixed-size vector for observation.
        
        Args:
            messages (List[AgentMessage]): The list of received messages.
            
        Returns:
            np.ndarray: Encoded observation vector.
        """
        if not self.enabled or not messages:
            return np.zeros(len(self.message_fields), dtype=np.float32)
            
        encoded = np.zeros(len(self.message_fields), dtype=np.float32)
        for msg in messages:
            vec = np.array([msg.content.get(f, 0.0) for f in self.message_fields], dtype=np.float32)
            encoded += vec
            
        encoded /= len(messages)
        return encoded
    
    def get_communication_overhead(self) -> Dict[str, float]:
        """Track communication metrics: num messages, total data size.
        
        Returns:
            Dict[str, float]: The communication metrics.
        """
        return {
            'num_messages': float(self.num_messages_sent),
            'total_data_size_bytes': float(self.total_data_size)
        }
    
    def reset(self) -> None:
        """Reset the communication manager's statistics."""
        self.num_messages_sent = 0
        self.total_data_size = 0.0
