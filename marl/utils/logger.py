"""
Experiment logging utilities.

Status: Implemented
"""
import os
import csv
import json
from typing import Dict, Optional, Any

class ExperimentLogger:
    """Structured experiment logging to CSV, JSON, and optionally TensorBoard.
    
    Status: Implemented
    """
    def __init__(self, log_dir: str, experiment_name: str = 'experiment',
                 use_tensorboard: bool = False):
        """Initialize the ExperimentLogger.
        
        Args:
            log_dir (str): Base directory for logs.
            experiment_name (str, optional): Name of the experiment. Defaults to 'experiment'.
            use_tensorboard (bool, optional): Whether to use TensorBoard. Defaults to False.
        """
        self.log_dir = os.path.join(log_dir, experiment_name)
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.csv_file = open(os.path.join(self.log_dir, "metrics.csv"), "w", newline="")
        self.csv_writer = None
        
        self.jsonl_file = open(os.path.join(self.log_dir, "logs.jsonl"), "w")
        
        self.tb_writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb_writer = SummaryWriter(log_dir=self.log_dir)
            except ImportError:
                print("TensorBoard not installed. Proceeding without it.")
    
    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a scalar value.
        
        Args:
            tag (str): The metric name.
            value (float): The metric value.
            step (int): The current step.
        """
        log_data = {"step": step, tag: value}
        self.jsonl_file.write(json.dumps(log_data) + "\n")
        self.jsonl_file.flush()
        
        if self.tb_writer:
            self.tb_writer.add_scalar(tag, value, step)
    
    def log_scalars(self, tag_value_dict: Dict[str, float], step: int) -> None:
        """Log multiple scalars at once.
        
        Args:
            tag_value_dict (Dict[str, float]): Dictionary of metrics to log.
            step (int): The current step.
        """
        log_data = {"step": step, **tag_value_dict}
        self.jsonl_file.write(json.dumps(log_data) + "\n")
        self.jsonl_file.flush()
        
        if self.tb_writer:
            for tag, value in tag_value_dict.items():
                self.tb_writer.add_scalar(tag, value, step)
    
    def log_episode(self, episode: int, metrics: Dict[str, float]) -> None:
        """Log all metrics for an episode. Writes to CSV.
        
        Args:
            episode (int): The episode number.
            metrics (Dict[str, float]): Dictionary of episode metrics.
        """
        row = {"episode": episode, **metrics}
        
        if self.csv_writer is None:
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=row.keys())
            self.csv_writer.writeheader()
            
        self.csv_writer.writerow(row)
        self.csv_file.flush()
        
        self.log_scalars(metrics, episode)
    
    def log_config(self, config: dict) -> None:
        """Save config as JSON.
        
        Args:
            config (dict): The configuration dictionary.
        """
        config_path = os.path.join(self.log_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)
    
    def log_text(self, tag: str, text: str, step: int) -> None:
        """Log text (e.g., for experiment notes).
        
        Args:
            tag (str): The tag for the text log.
            text (str): The text content.
            step (int): The current step.
        """
        log_data = {"step": step, "tag": tag, "text": text}
        self.jsonl_file.write(json.dumps(log_data) + "\n")
        self.jsonl_file.flush()
        
        if self.tb_writer:
            self.tb_writer.add_text(tag, text, step)
    
    def close(self) -> None:
        """Flush and close all writers."""
        if self.csv_file:
            self.csv_file.close()
        if self.jsonl_file:
            self.jsonl_file.close()
        if self.tb_writer:
            self.tb_writer.close()
    
    def get_log_dir(self) -> str:
        """Get the log directory.
        
        Returns:
            str: The path to the log directory.
        """
        return self.log_dir
