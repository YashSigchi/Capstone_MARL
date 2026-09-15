"""
Value prediction utilities.

Status: Placeholder / Baseline
"""
from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class ValuePredictor(ABC):
    """Generic interface for state value prediction.
    
    This is a PLACEHOLDER interface. The exact Matrix-V formulation
    will be implemented once its mathematical definition is finalized
    by the research team.
    
    Current implementation: simple linear baseline.
    Future implementation: Matrix-V formulation (TBD).
    
    Status: Placeholder
    """
    @abstractmethod
    def predict(self, state: np.ndarray) -> float:
        """Predict the value of a state."""
        pass
    
    @abstractmethod
    def update(self, states: np.ndarray, targets: np.ndarray) -> float:
        """Update predictor. Returns loss."""
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save predictor model."""
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """Load predictor model."""
        pass

class BaselineValuePredictor(ValuePredictor):
    """Simple linear value estimator.
    
    Uses a single linear layer: V(s) = w^T s + b
    Trained with MSE loss.
    
    Status: Baseline
    """
    def __init__(self, state_dim: int, lr: float = 1e-3):
        """Initialize the BaselineValuePredictor.
        
        Args:
            state_dim (int): Dimensionality of the state vector.
            lr (float, optional): Learning rate. Defaults to 1e-3.
        """
        self.state_dim = state_dim
        self.model = nn.Linear(state_dim, 1)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        
    def predict(self, state: np.ndarray) -> float:
        """Predict the value of a state.
        
        Args:
            state (np.ndarray): State array.
            
        Returns:
            float: Predicted value.
        """
        state_tensor = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            value = self.model(state_tensor).item()
        return value
        
    def update(self, states: np.ndarray, targets: np.ndarray) -> float:
        """Update predictor with states and targets.
        
        Args:
            states (np.ndarray): Batch of states.
            targets (np.ndarray): Batch of targets.
            
        Returns:
            float: Training loss.
        """
        states_tensor = torch.tensor(states, dtype=torch.float32)
        targets_tensor = torch.tensor(targets, dtype=torch.float32).unsqueeze(-1)
        
        self.optimizer.zero_grad()
        predictions = self.model(states_tensor)
        loss = self.loss_fn(predictions, targets_tensor)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
        
    def save(self, path: str) -> None:
        """Save the model weights.
        
        Args:
            path (str): File path.
        """
        torch.save(self.model.state_dict(), path)
        
    def load(self, path: str) -> None:
        """Load the model weights.
        
        Args:
            path (str): File path.
        """
        self.model.load_state_dict(torch.load(path))
