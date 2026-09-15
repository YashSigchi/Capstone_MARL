"""
Checkpoint management utilities.

Status: Implemented
"""
import os
import glob
import json
import torch
import shutil
from typing import Dict, Any, Optional
from datetime import datetime

class CheckpointManager:
    """Save and load model checkpoints with configuration.
    
    Saves: model state dicts, optimizer states, config, training step,
    best metric value.
    
    Status: Implemented
    """
    def __init__(self, checkpoint_dir: str, max_checkpoints: int = 5):
        """Initialize the CheckpointManager.
        
        Args:
            checkpoint_dir (str): Directory to save checkpoints.
            max_checkpoints (int, optional): Maximum number of checkpoints to keep. Defaults to 5.
        """
        self.checkpoint_dir = checkpoint_dir
        self.max_checkpoints = max_checkpoints
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def save(self, algorithm: Any, config: dict,
             step: int, metrics: Optional[Dict[str, float]] = None,
             is_best: bool = False) -> str:
        """Save checkpoint. Returns path.
        If is_best, also save as 'best_model.pt'.
        Always save as 'latest_model.pt'.
        Also save numbered checkpoint.
        
        Args:
            algorithm (Any): The MARL algorithm instance containing save method or state_dict.
            config (dict): Configuration dictionary.
            step (int): Current training step.
            metrics (Optional[Dict[str, float]], optional): Current metrics. Defaults to None.
            is_best (bool, optional): Whether this is the best model so far. Defaults to False.
            
        Returns:
            str: Path to the saved numbered checkpoint.
        """
        timestamp = datetime.now().isoformat()
        
        algorithm_state = {}
        if hasattr(algorithm, 'save'):
            algorithm_state = algorithm.save()
        elif hasattr(algorithm, 'state_dict'):
            algorithm_state = algorithm.state_dict()
            
        checkpoint = {
            'step': step,
            'config': config,
            'algorithm_state': algorithm_state,
            'metrics': metrics,
            'timestamp': timestamp,
        }
        
        numbered_path = os.path.join(self.checkpoint_dir, f"checkpoint_{step}.pt")
        latest_path = os.path.join(self.checkpoint_dir, "latest_model.pt")
        best_path = os.path.join(self.checkpoint_dir, "best_model.pt")
        
        torch.save(checkpoint, numbered_path)
        shutil.copyfile(numbered_path, latest_path)
        
        if is_best:
            shutil.copyfile(numbered_path, best_path)
            
        self._cleanup_old_checkpoints()
        
        return numbered_path
    
    def load(self, path: str) -> dict:
        """Load checkpoint dict.
        
        Args:
            path (str): Path to the checkpoint file.
            
        Returns:
            dict: The loaded checkpoint dictionary.
        """
        return torch.load(path)
    
    def load_best(self) -> dict:
        """Load best_model.pt.
        
        Returns:
            dict: The best model checkpoint dictionary.
        """
        best_path = os.path.join(self.checkpoint_dir, "best_model.pt")
        return self.load(best_path)
    
    def load_latest(self) -> dict:
        """Load latest_model.pt.
        
        Returns:
            dict: The latest model checkpoint dictionary.
        """
        latest_path = os.path.join(self.checkpoint_dir, "latest_model.pt")
        return self.load(latest_path)
    
    def _cleanup_old_checkpoints(self) -> None:
        """Remove oldest checkpoints beyond max_checkpoints."""
        checkpoint_files = glob.glob(os.path.join(self.checkpoint_dir, "checkpoint_*.pt"))
        
        def get_step(filepath: str) -> int:
            basename = os.path.basename(filepath)
            return int(basename.split('_')[1].split('.')[0])
            
        checkpoint_files.sort(key=get_step)
        
        if len(checkpoint_files) > self.max_checkpoints:
            for file_to_remove in checkpoint_files[:-self.max_checkpoints]:
                os.remove(file_to_remove)
