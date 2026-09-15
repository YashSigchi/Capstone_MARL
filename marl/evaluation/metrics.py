from dataclasses import dataclass, field
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

@dataclass
class EpisodeMetrics:
    """Raw physical/system metrics for one episode."""
    makespan: float = 0.0
    avg_task_latency: float = 0.0
    p95_latency: float = 0.0
    p99_latency: float = 0.0
    avg_queue_waiting_time: float = 0.0
    throughput: float = 0.0
    avg_cpu_utilization: float = 0.0
    avg_memory_utilization: float = 0.0
    resource_imbalance: float = 0.0
    total_energy: float = 0.0
    total_carbon: float = 0.0
    completed_tasks: float = 0.0
    unfinished_tasks: float = 0.0
    energy_per_completed_task: float = 0.0
    carbon_per_completed_task: float = 0.0
    sla_violation_rate: float = 0.0
    high_priority_sla_violations: float = 0.0
    avg_deadline_slack: float = 0.0
    task_completion_rate: float = 0.0
    task_rejection_rate: float = 0.0
    total_reward: float = 0.0
    reward_components: Dict[str, float] = field(default_factory=dict)

class MetricsTracker:
    """Tracks raw physical/system metrics across episodes.
    
    Normalized reward is logged but NOT used as primary evaluation evidence.
    
    Status: Implemented
    """
    def __init__(self):
        self._step_data: List[dict] = []
        self._episode_data: List[EpisodeMetrics] = []
        self._current_latencies: List[float] = []
        self._current_queue_waits: List[float] = []
        self._current_utilizations: List[List[float]] = []
        self._current_energy: float = 0.0
        self._current_carbon: float = 0.0
        self._current_reward: float = 0.0
        self._current_reward_components: Dict[str, float] = {}
        self._completed_tasks: int = 0
        self._rejected_tasks: int = 0
        self._total_tasks: int = 0
    
    def record_step(self, step_info: dict) -> None:
        """Record per-step data: latency, energy, utilizations, sla, etc."""
        self._step_data.append(step_info)
        
        if 'latency' in step_info:
            self._current_latencies.extend(step_info['latency'])
        if 'queue_wait' in step_info:
            self._current_queue_waits.extend(step_info['queue_wait'])
        if 'utilization' in step_info:
            self._current_utilizations.append(step_info['utilization'])
        
        self._current_energy += step_info.get('energy', 0.0)
        self._current_carbon += step_info.get('carbon', 0.0)
        self._current_reward += step_info.get('reward', 0.0)
        
        if 'reward_components' in step_info:
            for k, v in step_info['reward_components'].items():
                self._current_reward_components[k] = self._current_reward_components.get(k, 0.0) + v
                
        self._completed_tasks += step_info.get('completed_tasks', 0)
        self._rejected_tasks += step_info.get('rejected_tasks', 0)
        self._total_tasks += step_info.get('new_tasks', 0)
    
    def finalize_episode(self, total_steps: int) -> EpisodeMetrics:
        """Compute episode-level metrics and store."""
        metrics = EpisodeMetrics()
        metrics.makespan = float(total_steps)
        
        if self._current_latencies:
            metrics.avg_task_latency = float(np.mean(self._current_latencies))
            metrics.p95_latency = float(np.percentile(self._current_latencies, 95))
            metrics.p99_latency = float(np.percentile(self._current_latencies, 99))
            
        if self._current_queue_waits:
            metrics.avg_queue_waiting_time = float(np.mean(self._current_queue_waits))
            
        if self._total_tasks > 0:
            metrics.throughput = self._completed_tasks / total_steps if total_steps > 0 else 0
            metrics.task_completion_rate = self._completed_tasks / self._total_tasks
            metrics.task_rejection_rate = self._rejected_tasks / self._total_tasks
            
        if self._current_utilizations:
            utils = np.array(self._current_utilizations)
            metrics.avg_cpu_utilization = float(np.mean(utils))
            # simplified for now, assuming scalar util, can be expanded
            metrics.avg_memory_utilization = float(np.mean(utils))
            metrics.resource_imbalance = float(np.std(utils))
            
        metrics.total_energy = self._current_energy
        metrics.total_carbon = self._current_carbon
        metrics.total_reward = self._current_reward
        metrics.reward_components = dict(self._current_reward_components)
        
        self._episode_data.append(metrics)
        self._reset_episode_state()
        return metrics

    def _reset_episode_state(self):
        self._step_data.clear()
        self._current_latencies.clear()
        self._current_queue_waits.clear()
        self._current_utilizations.clear()
        self._current_energy = 0.0
        self._current_carbon = 0.0
        self._current_reward = 0.0
        self._current_reward_components.clear()
        self._completed_tasks = 0
        self._rejected_tasks = 0
        self._total_tasks = 0
    
    def get_summary(self, last_n: Optional[int] = None) -> Dict[str, float]:
        """Aggregated metrics (mean, std) over last_n episodes."""
        if not self._episode_data:
            return {}
            
        episodes = self._episode_data[-last_n:] if last_n else self._episode_data
        summary = {}
        df = pd.DataFrame([vars(e) for e in episodes])
        
        for col in df.columns:
            if col != 'reward_components':
                summary[f'{col}_mean'] = float(df[col].mean())
                summary[f'{col}_std'] = float(df[col].std())
                
        return summary
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert all episode metrics to pandas DataFrame."""
        return pd.DataFrame([vars(e) for e in self._episode_data])
    
    def to_csv(self, path: str) -> None:
        self.to_dataframe().to_csv(path, index=False)
        
    def to_json(self, path: str) -> None:
        self.to_dataframe().to_json(path, orient='records', indent=2)
        
    def reset(self) -> None:
        self._episode_data.clear()
        self._reset_episode_state()
